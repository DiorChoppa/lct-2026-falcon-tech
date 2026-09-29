use lct_inference::Embeddings;
use lct_offline::{read_rows, write_submission};
use std::{fs, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

fn root() -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/offline_driver_checks")
        .join(format!("rust_unit_{}_{}", std::process::id(), nonce));
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn outputs_preserve_ties_f64_rejection_and_raw_vectors() {
    let root = root();
    let queries = vec!["q0".to_string(), "q1".to_string()];
    let gallery = (0..12).map(|i| format!("g{i}")).collect::<Vec<_>>();
    let mut data = vec![1_f32, 0., 0., 1.];
    for i in 0..12 { data.extend(if i == 11 { [0., 1.] } else { [1., 0.] }); }
    let matrix = Embeddings { rows: 14, dimensions: 2, data };
    let rejected = root.join("reject_all");
    write_submission(&queries, &gallery, &matrix, f64::from_bits(1_f64.to_bits() + 1), &rejected).unwrap();
    let rows = csv::ReaderBuilder::new().has_headers(false).from_path(rejected.join("submission.csv")).unwrap()
        .records().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), 11);
    assert_eq!(rows[0].iter().skip(1).collect::<Vec<_>>(), (0..10).map(|i| gallery[i].as_str()).collect::<Vec<_>>());
    assert_eq!(csv::Reader::from_path(rejected.join("candidates.csv")).unwrap().records().count(), 0);
    assert_eq!(fs::read_dir(&rejected).unwrap().count(), 3);
    let npy = fs::read(rejected.join("embeddings.npy")).unwrap();
    assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
    let start = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
    assert_eq!(start % 16, 0);
    let bytes = matrix.data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(&npy[start..], bytes);
    assert!(write_submission(&queries, &gallery, &matrix, 1., &rejected).is_err());
    let accepted = root.join("inclusive");
    write_submission(&queries, &gallery, &matrix, 1., &accepted).unwrap();
    assert_eq!(csv::Reader::from_path(accepted.join("candidates.csv")).unwrap().records().count(), 2);
    let invalid = root.join("invalid");
    let mut nonfinite = matrix;
    nonfinite.data[0] = f32::NAN;
    assert!(write_submission(&queries, &gallery, &nonfinite, 1., &invalid).is_err());
    assert!(!invalid.exists());
}

#[test]
fn canonical_csv_ignores_labels_and_rejects_invalid_rows() {
    let root = root();
    let valid = root.join("valid.csv");
    fs::write(&valid, "image_id,x,y,w,h,vehicle_id,camera_id\nq,0,1,10,20,ignored,also_ignored\n").unwrap();
    let rows = read_rows(&valid).unwrap();
    assert_eq!(rows[0].image_id, "q");
    assert_eq!(rows[0].h, 20);
    for (i, body) in [
        "image_id,x,y,w,h\nq,0,0,1,1\nq,0,0,1,1\n",
        "image_id,x,y,w,h\nq,0,0,0,1\n",
        "image_id,x,y,w,h\n../escape,0,0,1,1\n",
        "image_id,x,y,w,h\nq,0,0,1.5,1\n",
        "image_id,x,y,w,h\nq,0,0,1\n",
        "image_id,x,y,w,w,h\nq,0,0,1,1,1\n",
        "image_id,x,y,w\nq,0,0,1\n",
    ].iter().enumerate() {
        let path = root.join(format!("invalid_{i}.csv"));
        fs::write(&path, body).unwrap();
        assert!(read_rows(&path).is_err(), "case {i}");
    }
}
