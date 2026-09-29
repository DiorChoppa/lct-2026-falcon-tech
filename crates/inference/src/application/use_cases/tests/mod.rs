pub mod fixtures;

use std::sync::Arc;

use common::BBox;
use image::imageops::FilterType;
use image::{codecs::png::PngEncoder, ImageEncoder, RgbImage};
use ndarray::{Array4, Axis};

use crate::application::repositories::EmbedOutput;
use crate::application::use_cases::helpers::{
    decode_crop_resize, l2_normalize, resize_filter, write_normalized,
};
use crate::application::use_cases::UseCases;
use crate::domain::{Crop, ModelManifest};
use crate::test_utils::mock_embedding_repository::MockEmbeddingRepository;
use fixtures::{stub_crop, test_manifest, test_manifest_with_patches};

fn encode_png(pixels: &[[u8; 3]], w: u32, h: u32) -> Vec<u8> {
    let mut img = RgbImage::new(w, h);
    for (i, px) in pixels.iter().enumerate() {
        img.put_pixel(i as u32 % w, i as u32 / w, image::Rgb(*px));
    }
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&img, w, h, image::ExtendedColorType::Rgb8)
        .unwrap();
    out
}

#[test]
fn resize_filter_maps_manifest_values() {
    assert!(matches!(
        resize_filter("bilinear").unwrap(),
        FilterType::Triangle
    ));
    assert!(matches!(
        resize_filter("bicubic").unwrap(),
        FilterType::CatmullRom
    ));
    assert!(resize_filter("nearest").is_err());
}

#[test]
fn decode_crop_resize_crops_and_resizes_to_target() {
    // 2x2: [red, green; blue, white], crop top-left 1x1 (red), resize to 4x4.
    let png = encode_png(
        &[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
        2,
        2,
    );
    let crop = Crop {
        image: png.into(),
        bbox: BBox {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        },
    };

    let resized = decode_crop_resize(&crop, 4, 4, FilterType::Triangle).unwrap();

    assert_eq!((4, 4), resized.dimensions());
    for pixel in resized.pixels() {
        assert_eq!(&[255, 0, 0], &pixel.0);
    }
}

#[test]
fn decode_crop_resize_clamps_out_of_frame_bbox() {
    let png = encode_png(&[[10, 20, 30]], 1, 1);
    let crop = Crop {
        image: png.into(),
        bbox: BBox {
            x: 5,
            y: 5,
            w: 10,
            h: 10,
        },
    };

    let resized = decode_crop_resize(&crop, 2, 2, FilterType::Triangle).unwrap();

    assert_eq!((2, 2), resized.dimensions());
}

#[test]
fn write_normalized_applies_mean_std_per_channel() {
    let img = {
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 128]));
        img
    };
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 1.0, 0.25];
    let mut batch = Array4::<f32>::zeros((1, 3, 1, 1));

    write_normalized(&img, &mean, &std, batch.index_axis_mut(Axis(0), 0));

    assert!((batch[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((batch[[0, 1, 0, 0]] - (-0.5)).abs() < 1e-6);
    assert!((batch[[0, 2, 0, 0]] - ((128.0 / 255.0 - 0.5) / 0.25)).abs() < 1e-6);
}

#[test]
fn l2_normalize_produces_unit_vector() {
    let mut v = vec![3.0, 4.0];
    l2_normalize(&mut v);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn l2_normalize_leaves_zero_vector_untouched() {
    let mut v = vec![0.0, 0.0];
    l2_normalize(&mut v);
    assert_eq!(vec![0.0, 0.0], v);
}

#[test]
fn embed_returns_empty_for_no_crops_without_calling_repository() {
    let repository = Arc::new(MockEmbeddingRepository::default());
    let use_cases = UseCases::new(test_manifest(), repository.clone());

    let result = use_cases.embed(&[], false).unwrap();

    assert!(result.is_empty());
    assert_eq!(0, repository.received_batch_count());
}

#[test]
fn embed_forwards_preprocessed_batch_to_repository() {
    let repository = Arc::new(MockEmbeddingRepository::new(vec![vec![0.0; 3]]));
    let use_cases = UseCases::new(test_manifest(), repository.clone());

    use_cases.embed(&[stub_crop()], false).unwrap();

    assert_eq!(vec![vec![1, 3, 4, 4]], repository.received_batch_shapes());
    assert_eq!(vec![false], repository.received_with_patches());
}

#[test]
fn embed_l2_normalizes_when_manifest_not_normalized() {
    let manifest = test_manifest();
    assert!(!manifest.l2_normalized);
    let repository = Arc::new(MockEmbeddingRepository::new(vec![vec![3.0, 4.0, 0.0]]));
    let use_cases = UseCases::new(manifest, repository);

    let result = use_cases.embed(&[stub_crop()], false).unwrap();

    assert_eq!(vec![0.6, 0.8, 0.0], result[0].embedding);
    assert_eq!(None, result[0].patches);
}

#[test]
fn embed_passes_through_when_manifest_already_normalized() {
    let manifest = ModelManifest {
        l2_normalized: true,
        ..test_manifest()
    };
    let repository = Arc::new(MockEmbeddingRepository::new(vec![vec![3.0, 4.0, 0.0]]));
    let use_cases = UseCases::new(manifest, repository);

    let result = use_cases.embed(&[stub_crop()], false).unwrap();

    assert_eq!(vec![3.0, 4.0, 0.0], result[0].embedding);
}

#[test]
fn embed_rejects_with_patches_when_model_has_no_patches_output() {
    let use_cases = UseCases::new(
        test_manifest(),
        Arc::new(MockEmbeddingRepository::default()),
    );

    let err = use_cases.embed(&[stub_crop()], true).unwrap_err();

    assert!(err.to_string().contains("no patches output"));
}

#[test]
fn embed_returns_and_normalizes_patches_when_model_supports_them() {
    let manifest = test_manifest_with_patches();
    let repository = Arc::new(MockEmbeddingRepository::with_outputs(vec![EmbedOutput {
        embedding: vec![3.0, 4.0, 0.0],
        // Two patch tokens of dim 2, each needing its own L2-normalization.
        patches: Some(vec![3.0, 4.0, 1.0, 0.0]),
    }]));
    let use_cases = UseCases::new(manifest, repository.clone());

    let result = use_cases.embed(&[stub_crop()], true).unwrap();

    assert_eq!(vec![true], repository.received_with_patches());
    assert_eq!(vec![0.6, 0.8, 1.0, 0.0], result[0].patches.clone().unwrap());
}

#[test]
fn info_maps_manifest_fields() {
    let use_cases = UseCases::new(
        test_manifest(),
        Arc::new(MockEmbeddingRepository::default()),
    );

    let info = use_cases.info();

    assert_eq!("test", info.model_name);
    assert_eq!(3, info.dim);
    assert_eq!(4, info.input_height);
    assert_eq!(4, info.input_width);
    assert!(!info.supports_patches);
}

#[test]
fn info_reports_patch_support_from_manifest() {
    let use_cases = UseCases::new(
        test_manifest_with_patches(),
        Arc::new(MockEmbeddingRepository::default()),
    );

    let info = use_cases.info();

    assert!(info.supports_patches);
    assert_eq!(1, info.patch_grid_h);
    assert_eq!(2, info.patch_grid_w);
    assert_eq!(2, info.patch_dim);
}
