use anyhow::{Result, bail};
use lct_inference::{BBox, Embeddings, Input};
use lct_offline::extract_separately;
use std::path::PathBuf;

fn observations(start: i64, count: usize) -> Vec<Input> {
    (0..count).map(|offset| Input {
        path: PathBuf::from(format!("{}.jpg", start + offset as i64)),
        bbox: BBox { x: start + offset as i64, y: 0, w: 1, h: 1 },
    }).collect()
}

#[test]
fn gallery_batches_and_vectors_do_not_depend_on_query_count() -> Result<()> {
    let gallery = observations(1000, 750);
    let mut reference = None;
    for query_count in [1, 31, 32, 33, 215] {
        let query = observations(0, query_count);
        let mut batches = Vec::new();
        let matrix = extract_separately(&query, &gallery, 32, |chunk| {
            let ids: Vec<_> = chunk.iter().map(|input| input.bbox.x).collect();
            batches.push(ids.clone());
            // Deliberately batch-size-dependent output exposes the original bug.
            Ok(Embeddings { rows: chunk.len(), dimensions: 2,
                data: ids.iter().flat_map(|id| [*id as f32, chunk.len() as f32]).collect() })
        })?;
        assert_eq!(matrix.rows, query_count + 750);
        assert_eq!(batches[0].len(), query_count.min(32)); // No query padding.
        assert!(batches.iter().all(|chunk| chunk.iter().all(|id| *id < 1000)
            || chunk.iter().all(|id| *id >= 1000)));
        let gallery_batches: Vec<_> = batches.into_iter().filter(|chunk| chunk[0] >= 1000).collect();
        assert_eq!(gallery_batches.last().unwrap().len(), 14);
        let gallery_values = matrix.data[query_count * 2..].to_vec();
        let current = (gallery_batches, gallery_values);
        if let Some(expected) = &reference { assert_eq!(&current, expected); }
        else { reference = Some(current); }
        let row_ids: Vec<_> = matrix.data.chunks_exact(2).map(|row| row[0] as i64).collect();
        assert_eq!(row_ids, (0..query_count as i64).chain(1000..1750).collect::<Vec<_>>());
    }
    Ok(())
}

#[test]
fn batch_one_remains_batch_one_for_both_roles() -> Result<()> {
    let query = observations(0, 3);
    let gallery = observations(1000, 11);
    let mut calls = 0;
    let matrix = extract_separately(&query, &gallery, 1, |chunk| {
        assert_eq!(chunk.len(), 1);
        calls += 1;
        Ok(Embeddings { rows: 1, dimensions: 1, data: vec![chunk[0].bbox.x as f32] })
    })?;
    assert_eq!(calls, 14);
    assert_eq!(matrix.rows, 14);
    Ok(())
}

#[test]
fn invalid_batches_and_extraction_errors_stop_before_gallery() {
    let query = observations(0, 2);
    let gallery = observations(1000, 10);
    let mut calls = 0;
    let error = extract_separately(&query, &gallery, 32, |_| {
        calls += 1;
        bail!("query decode failed")
    }).unwrap_err();
    assert_eq!(error.to_string(), "query decode failed");
    assert_eq!(calls, 1);
    assert!(extract_separately(&query, &gallery, 0, |_| unreachable!()).is_err());
    assert!(extract_separately(&query, &gallery, 32, |chunk|
        Ok(Embeddings { rows: chunk.len(), dimensions: 2, data: vec![0.0] })).is_err());
}
