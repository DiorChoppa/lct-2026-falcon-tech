use image::imageops::FilterType;
use image::RgbImage;
use ndarray::{Array4, ArrayViewMut3, Axis};

use crate::domain::{Crop, ModelManifest};

pub fn preprocess_batch(crops: &[Crop], manifest: &ModelManifest) -> anyhow::Result<Array4<f32>> {
    let height = manifest.input_height as usize;
    let width = manifest.input_width as usize;
    let filter = resize_filter(&manifest.resize)?;

    let mut batch = Array4::<f32>::zeros((crops.len(), 3, height, width));
    for (i, crop) in crops.iter().enumerate() {
        let resized = decode_crop_resize(crop, width as u32, height as u32, filter)?;
        write_normalized(
            &resized,
            &manifest.mean,
            &manifest.std,
            batch.index_axis_mut(Axis(0), i),
        );
    }
    Ok(batch)
}

/// "bilinear" | "bicubic" — manifest contract, see domain::ModelManifest.
pub fn resize_filter(resize: &str) -> anyhow::Result<FilterType> {
    match resize {
        "bilinear" => Ok(FilterType::Triangle),
        "bicubic" => Ok(FilterType::CatmullRom),
        other => anyhow::bail!("unknown resize in manifest: {other}"),
    }
}

pub fn decode_crop_resize(
    crop: &Crop,
    width: u32,
    height: u32,
    filter: FilterType,
) -> anyhow::Result<RgbImage> {
    let decoded = image::load_from_memory(&crop.image)?.to_rgb8();
    let (frame_w, frame_h) = decoded.dimensions();
    let bbox = crop.bbox.clamp(frame_w, frame_h);
    anyhow::ensure!(
        bbox.w > 0 && bbox.h > 0,
        "bbox is empty after clamping to frame {frame_w}x{frame_h}"
    );

    let cropped = image::imageops::crop_imm(&decoded, bbox.x, bbox.y, bbox.w, bbox.h).to_image();
    Ok(image::imageops::resize(&cropped, width, height, filter))
}

pub fn write_normalized(
    image: &RgbImage,
    mean: &[f32; 3],
    std: &[f32; 3],
    mut dst: ArrayViewMut3<f32>,
) {
    for (x, y, pixel) in image.enumerate_pixels() {
        for c in 0..3 {
            let value = pixel.0[c] as f32 / 255.0;
            dst[[c, y as usize, x as usize]] = (value - mean[c]) / std[c];
        }
    }
}

pub fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}
