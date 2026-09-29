//! Parity test for Python vs Rust inference: the reference crop/embedding in
//! `ml/parity/` must match within tolerance. `#[ignore]`d until those files land.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::BBox;
use image::GenericImageView;
use inference::{Crop, ModelManifest, OrtEmbeddingRepository, UseCases};
// ndarray-npy depends on ndarray 0.16, separate from our main ndarray 0.17
// (see the ndarray-for-npy comment in Cargo.toml).
use ndarray_for_npy::Array1;
use ndarray_npy::ReadNpyExt;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
#[ignore = "waiting on the reference files in ml/parity/ (contract #1)"]
fn embed_matches_python_reference_within_tolerance() {
    let root = workspace_root();
    let manifest_path = root.join("models/model.json");
    let crop_path = root.join("ml/parity/crop.png");
    let embedding_path = root.join("ml/parity/embedding.npy");

    let manifest = ModelManifest::load(&manifest_path)
        .unwrap_or_else(|err| panic!("{}: {err}", manifest_path.display()));
    let model_dir = manifest_path.parent().expect("manifest_path has a parent");
    let repository = Arc::new(OrtEmbeddingRepository::new(&manifest, model_dir));
    let use_cases = UseCases::new(manifest, repository);

    let image_bytes =
        std::fs::read(&crop_path).unwrap_or_else(|err| panic!("{}: {err}", crop_path.display()));
    let (w, h) = image::load_from_memory(&image_bytes)
        .unwrap_or_else(|err| panic!("decode {}: {err}", crop_path.display()))
        .dimensions();
    let crop = Crop {
        image: image_bytes.into(),
        bbox: BBox { x: 0, y: 0, w, h },
    };

    let reference_file = File::open(&embedding_path)
        .unwrap_or_else(|err| panic!("{}: {err}", embedding_path.display()));
    let reference = Array1::<f32>::read_npy(reference_file)
        .expect("ml/parity/embedding.npy is a valid .npy float32 file");

    let got = use_cases.embed(&[crop], false).expect("embed");
    let got = got.first().expect("one crop -> one vector");

    let similarity = common::cosine(
        &got.embedding,
        reference
            .as_slice()
            .expect("reference is a contiguous array"),
    );
    assert!(
        similarity >= 0.999,
        "Python/Rust parity: cosine = {similarity}, expected >= 0.999"
    );
}
