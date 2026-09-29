//! End-to-end smoke test of the decode -> crop -> resize -> normalize -> ORT ->
//! L2-normalize pipeline, against the synthetic `tests/fixtures/smoke_model.onnx`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::BBox;
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};
use inference::{Crop, ModelManifest, OrtEmbeddingRepository, UseCases};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn manifest() -> ModelManifest {
    ModelManifest {
        name: "smoke".into(),
        version: "0".into(),
        file: "smoke_model.onnx".into(),
        input_name: "input".into(),
        output_name: "embedding".into(),
        input_height: 4,
        input_width: 4,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        resize: "bilinear".into(),
        dim: 3,
        l2_normalized: false,
        patches_output_name: None,
        patch_grid_h: None,
        patch_grid_w: None,
        patch_dim: None,
    }
}

fn use_cases() -> UseCases {
    let manifest = manifest();
    let repository = Arc::new(OrtEmbeddingRepository::new(&manifest, &fixtures_dir()));
    UseCases::new(manifest, repository)
}

fn solid_png(rgb: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(4, 4, image::Rgb(rgb));
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&img, 4, 4, ExtendedColorType::Rgb8)
        .unwrap();
    out
}

fn l2_normalized(mut v: [f32; 3]) -> [f32; 3] {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[test]
fn embed_runs_full_pipeline_on_synthetic_model() {
    // 51/102/204 -> 0.2/0.4/0.8 after normalization (mean=0, std=1).
    let png = solid_png([51, 102, 204]);
    let crop = Crop {
        image: Arc::from(png.into_boxed_slice()),
        bbox: BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
    };

    let use_cases = use_cases();

    let result = use_cases
        .embed(&[crop.clone(), crop], false)
        .expect("embed failed");

    assert_eq!(2, result.len());
    let expected = l2_normalized([0.2, 0.4, 0.8]);
    for output in result {
        let vector = output.embedding;
        assert_eq!(3, vector.len());
        for (got, want) in vector.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "got {vector:?}, want {expected:?}"
            );
        }
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit L2 norm, got {norm}"
        );
        assert_eq!(None, output.patches);
    }
}

#[test]
fn embed_returns_empty_for_no_crops() {
    let use_cases = use_cases();

    let result = use_cases.embed(&[], false).unwrap();

    assert!(result.is_empty());
}

#[test]
fn embed_returns_patches_from_second_onnx_output_when_requested() {
    // smoke_model.onnx's "patches" output is ReduceMax (vs "embedding"'s
    // ReduceMean) over the same input — a real second output from one ORT
    // session, not a mock.
    let manifest = ModelManifest {
        patches_output_name: Some("patches".into()),
        patch_grid_h: Some(1),
        patch_grid_w: Some(1),
        patch_dim: Some(3),
        ..manifest()
    };
    let repository = Arc::new(OrtEmbeddingRepository::new(&manifest, &fixtures_dir()));
    let use_cases = UseCases::new(manifest, repository);

    // Two-tone 4x4 crop so mean (embedding) and max (patches) differ per channel.
    let mut img = RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            let rgb = if y < 2 { [10, 20, 30] } else { [90, 60, 240] };
            img.put_pixel(x, y, image::Rgb(rgb));
        }
    }
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&img, 4, 4, ExtendedColorType::Rgb8)
        .unwrap();
    let crop = Crop {
        image: Arc::from(png.into_boxed_slice()),
        bbox: BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
    };

    let result = use_cases.embed(&[crop], true).expect("embed failed");

    assert_eq!(1, result.len());
    let embedding = &result[0].embedding;
    let patches = result[0].patches.as_ref().expect("patches were requested");
    assert_eq!(3, embedding.len());
    assert_eq!(3, patches.len());
    assert_ne!(
        embedding, patches,
        "mean and max reductions must differ for a two-tone crop"
    );

    let expected_embedding = l2_normalized([50.0 / 255.0, 40.0 / 255.0, 135.0 / 255.0]);
    let expected_patches = l2_normalized([90.0 / 255.0, 60.0 / 255.0, 240.0 / 255.0]);
    for (got, want) in embedding.iter().zip(expected_embedding.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "embedding: got {embedding:?}, want {expected_embedding:?}"
        );
    }
    for (got, want) in patches.iter().zip(expected_patches.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "patches: got {patches:?}, want {expected_patches:?}"
        );
    }
}

#[test]
fn embed_rejects_with_patches_when_manifest_has_no_patches_output() {
    let use_cases = use_cases();
    let crop = Crop {
        image: Arc::from(solid_png([1, 2, 3]).into_boxed_slice()),
        bbox: BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
    };

    let err = use_cases.embed(&[crop], true).unwrap_err();

    assert!(err.to_string().contains("no patches output"));
}
