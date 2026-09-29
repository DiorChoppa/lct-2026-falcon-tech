use image::{codecs::png::PngEncoder, ImageEncoder, RgbImage};

use crate::domain::{Crop, ModelManifest};

pub fn test_manifest() -> ModelManifest {
    ModelManifest {
        name: "test".into(),
        version: "0".into(),
        file: "model.onnx".into(),
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

/// Same as `test_manifest()`, but with a configured patches output so
/// `with_patches` requests succeed.
pub fn test_manifest_with_patches() -> ModelManifest {
    ModelManifest {
        patches_output_name: Some("patches".into()),
        patch_grid_h: Some(1),
        patch_grid_w: Some(2),
        patch_dim: Some(2),
        ..test_manifest()
    }
}

/// Valid 4x4 crop compatible with `test_manifest()` (same input size).
pub fn stub_crop() -> Crop {
    let img = RgbImage::from_pixel(4, 4, image::Rgb([120, 60, 200]));
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&img, 4, 4, image::ExtendedColorType::Rgb8)
        .unwrap();

    Crop {
        image: png.into(),
        bbox: common::BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
    }
}
