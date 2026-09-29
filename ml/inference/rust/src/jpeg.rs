//! Decode a JPEG bbox with a chroma halo, preserving full-decode RGB pixels.
use crate::{BBox, Error, Result, checked_bbox};
use image::{RgbImage, imageops};
use std::ffi::CStr;
use turbojpeg::raw;

struct Decoder(raw::tjhandle);

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: this handle is non-null, uniquely owned, and destroyed once.
        unsafe { raw::tj3Destroy(self.0) };
    }
}

impl Decoder {
    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: TurboJPEG owns a NUL-terminated error string for this live handle.
        let message = unsafe { CStr::from_ptr(raw::tj3GetErrorStr(self.0)) };
        Err(Error::Invalid(format!(
            "JPEG decode: {}",
            message.to_string_lossy()
        )))
    }
}

pub(crate) fn crop(bytes: &[u8], bbox: BBox) -> Result<RgbImage> {
    // SAFETY: initialization takes no pointers; null is checked before use.
    let handle = unsafe { raw::tj3Init(raw::TJINIT_TJINIT_DECOMPRESS as i32) };
    if handle.is_null() {
        return Err(Error::Invalid("Cannot initialize JPEG decoder".into()));
    }
    let decoder = Decoder(handle);
    // SAFETY: the borrowed input remains live throughout each synchronous call.
    decoder.check(unsafe { raw::tj3DecompressHeader(handle, bytes.as_ptr(), bytes.len() as _) })?;
    // SAFETY: all parameters are read from the successfully parsed header.
    let (width, height, subsamp, lossless) = unsafe {
        (
            raw::tj3Get(handle, raw::TJPARAM_TJPARAM_JPEGWIDTH as i32),
            raw::tj3Get(handle, raw::TJPARAM_TJPARAM_JPEGHEIGHT as i32),
            raw::tj3Get(handle, raw::TJPARAM_TJPARAM_SUBSAMP as i32),
            raw::tj3Get(handle, raw::TJPARAM_TJPARAM_LOSSLESS as i32),
        )
    };
    if width <= 0 || height <= 0 {
        return Err(Error::Invalid("Invalid JPEG dimensions".into()));
    }
    let (x, y, w, h) = checked_bbox(bbox, width as u32, height as u32)?;
    // Lossless and unusual subsampling retain the existing full-decode path.
    if lossless != 0 || !(0..7).contains(&subsamp) {
        let decoded = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)?;
        let frame = RgbImage::from_raw(width as u32, height as u32, decoded.pixels)
            .ok_or_else(|| Error::Invalid("Invalid JPEG buffer".into()))?;
        return Ok(imageops::crop_imm(&frame, x, y, w, h).to_image());
    }
    // turbojpeg.h's static const tjMCUWidth is not an exported linker symbol.
    let mcu = [8_u32, 16, 16, 8, 8, 32, 8][subsamp as usize];
    // Keep one complete iMCU on both sides for identical fancy chroma upsampling.
    let left = x.saturating_sub(mcu) / mcu * mcu;
    let right = (x + w + mcu).min(width as u32);
    let region = raw::tjregion {
        x: left as i32,
        y: y as i32,
        w: (right - left) as i32,
        h: h as i32,
    };
    // SAFETY: the validated region is inside the frame and its left edge is iMCU-aligned.
    decoder.check(unsafe { raw::tj3SetCroppingRegion(handle, region) })?;
    let count = (region.w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Invalid("JPEG crop allocation overflow".into()))?;
    let mut pixels = vec![0_u8; count];
    // SAFETY: the initialized RGB buffer holds exactly region.w * region.h * 3 bytes;
    // pitch=0 selects that packed width. No pointers outlive this call.
    decoder.check(unsafe {
        raw::tj3Decompress8(
            handle,
            bytes.as_ptr(),
            bytes.len() as _,
            pixels.as_mut_ptr(),
            0,
            raw::TJPF_TJPF_RGB,
        )
    })?;
    let frame = RgbImage::from_raw(region.w as u32, h, pixels)
        .ok_or_else(|| Error::Invalid("Invalid JPEG crop buffer".into()))?;
    Ok(imageops::crop_imm(&frame, x - left, 0, w, h).to_image())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires LCT_PARITY_CSV with original local images and LCT_PARITY_REPORT"]
    fn manifest_pixels_match_legacy_pipeline() {
        #[derive(serde::Deserialize)]
        struct Row {
            image_id: String,
            path: String,
            x: i64,
            y: i64,
            w: i64,
            h: i64,
        }
        let manifest = std::env::var("LCT_PARITY_CSV").unwrap();
        let mut reader = csv::Reader::from_path(manifest).unwrap();
        let mut legacy_ms = Vec::new();
        let mut optimized_ms = Vec::new();
        for record in reader.deserialize::<Row>() {
            let row = record.unwrap();
            let bbox = BBox {
                x: row.x,
                y: row.y,
                w: row.w,
                h: row.h,
            };
            let legacy = || {
                let start = std::time::Instant::now();
                let frame = crate::decode_rgb(std::path::Path::new(&row.path)).unwrap();
                let crop = imageops::crop_imm(
                    &frame,
                    row.x as u32,
                    row.y as u32,
                    row.w as u32,
                    row.h as u32,
                )
                .to_image();
                let square = crate::resize::reference_bicubic(&crop, 256, 256);
                (crop, square, start.elapsed().as_secs_f64() * 1000.0)
            };
            let optimized = || {
                let start = std::time::Instant::now();
                let bytes = std::fs::read(&row.path).unwrap();
                let crop = crop(&bytes, bbox).unwrap();
                let square = crate::resize::bicubic(&crop, 256, 256);
                (crop, square, start.elapsed().as_secs_f64() * 1000.0)
            };
            // Alternate order to share warm filesystem/cache effects fairly.
            let (old, new) = if legacy_ms.len() % 2 == 0 {
                (legacy(), optimized())
            } else {
                let new = optimized();
                (legacy(), new)
            };
            assert_eq!(old.0.as_raw(), new.0.as_raw(), "JPEG crop {}", row.image_id);
            assert_eq!(old.1.as_raw(), new.1.as_raw(), "resize {}", row.image_id);
            legacy_ms.push(old.2);
            optimized_ms.push(new.2);
        }
        assert!(!legacy_ms.is_empty());
        legacy_ms.sort_by(f64::total_cmp);
        optimized_ms.sort_by(f64::total_cmp);
        let report = serde_json::json!({"rows": legacy_ms.len(), "crop_mismatches": 0,
            "resize_mismatches": 0, "legacy_median_ms": legacy_ms[legacy_ms.len()/2],
            "optimized_median_ms": optimized_ms[optimized_ms.len()/2],
            "boundary": "file read, JPEG decode, bbox crop, RGB bicubic resize; no normalization or forward",
            "normalization": "unchanged arithmetic over byte-identical RGB pixels"});
        std::fs::write(
            std::env::var("LCT_PARITY_REPORT").unwrap(),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("{report}");
    }

    #[test]
    fn cropped_decode_matches_full_rgb_at_chroma_and_frame_edges() {
        let pixels: Vec<u8> = (0..97 * 73 * 3)
            .map(|i| ((i * 37 + i / 11) % 256) as u8)
            .collect();
        for subsamp in [
            turbojpeg::Subsamp::None,
            turbojpeg::Subsamp::Sub2x1,
            turbojpeg::Subsamp::Sub2x2,
            turbojpeg::Subsamp::Gray,
        ] {
            let source = turbojpeg::Image {
                pixels: pixels.as_slice(),
                width: 97,
                height: 73,
                pitch: 97 * 3,
                format: turbojpeg::PixelFormat::RGB,
            };
            let jpeg = turbojpeg::compress(source, 92, subsamp).unwrap();
            let full = turbojpeg::decompress(&jpeg, turbojpeg::PixelFormat::RGB).unwrap();
            let full = RgbImage::from_raw(97, 73, full.pixels).unwrap();
            for (x, y, w, h) in [
                (0, 0, 97, 73),
                (1, 1, 1, 1),
                (31, 7, 33, 29),
                (79, 57, 18, 16),
                (16, 16, 1, 1),
            ] {
                let got = crop(&jpeg, BBox { x, y, w, h }).unwrap();
                assert_eq!(
                    got,
                    imageops::crop_imm(&full, x as u32, y as u32, w as u32, h as u32).to_image()
                );
            }
        }
    }
}
