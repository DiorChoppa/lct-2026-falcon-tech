use std::io::Cursor;

use crate::BBox;

/// Cuts `bbox` out of `image_bytes` and re-encodes as JPEG at `quality`. The
/// one crop implementation shared by gallery/search when they write
/// gallery-crops/query-crops, so a stored crop matches byte-for-byte what
/// inference would decode for embedding.
pub fn crop_jpeg(image_bytes: &[u8], bbox: BBox, quality: u8) -> anyhow::Result<Vec<u8>> {
    let decoded = image::load_from_memory(image_bytes)?;
    let bbox = bbox.clamp(decoded.width(), decoded.height());
    anyhow::ensure!(
        bbox.w > 0 && bbox.h > 0,
        "bbox is empty after clamping to frame {}x{}",
        decoded.width(),
        decoded.height()
    );

    let cropped = decoded.crop_imm(bbox.x, bbox.y, bbox.w, bbox.h).to_rgb8();
    let mut buf = Cursor::new(Vec::new());
    cropped.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
        &mut buf, quality,
    ))?;
    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};

    use super::*;

    fn solid_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let img = RgbImage::from_pixel(w, h, image::Rgb(rgb));
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(&img, w, h, ExtendedColorType::Rgb8)
            .unwrap();
        out
    }

    #[test]
    fn crop_jpeg_produces_decodable_image_of_bbox_size() {
        let png = solid_png(8, 8, [10, 20, 30]);
        let bbox = BBox {
            x: 1,
            y: 1,
            w: 4,
            h: 3,
        };

        let jpeg = crop_jpeg(&png, bbox, 95).unwrap();

        let decoded = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((4, 3), (decoded.width(), decoded.height()));
    }

    #[test]
    fn crop_jpeg_clamps_bbox_to_frame() {
        let png = solid_png(4, 4, [1, 2, 3]);
        let bbox = BBox {
            x: 2,
            y: 2,
            w: 10,
            h: 10,
        };

        let jpeg = crop_jpeg(&png, bbox, 95).unwrap();

        let decoded = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((2, 2), (decoded.width(), decoded.height()));
    }

    #[test]
    fn crop_jpeg_rejects_zero_size_bbox() {
        let png = solid_png(4, 4, [1, 2, 3]);
        let bbox = BBox {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };

        let err = crop_jpeg(&png, bbox, 95).unwrap_err();

        assert!(err.to_string().contains("empty after clamping"));
    }

    #[test]
    fn crop_jpeg_rejects_undecodable_bytes() {
        let err = crop_jpeg(
            b"not an image",
            BBox {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            95,
        )
        .unwrap_err();

        assert!(!err.to_string().is_empty());
    }
}
