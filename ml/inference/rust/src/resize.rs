//! Pillow 12.3 bicubic RGB resizing: antialias support, 22-bit coefficients,
//! horizontal then vertical 8-bit clipping. See THIRD_PARTY_NOTICES.md.
use image::RgbImage;

const PRECISION: u32 = 22;

struct Kernel {
    start: u32,
    weights: Vec<i32>,
}

fn cubic(value: f64) -> f64 {
    let x = value.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

fn coefficients(input: u32, output: u32) -> Vec<Kernel> {
    let scale = f64::from(input) / f64::from(output);
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let inverse = 1.0 / filter_scale;
    (0..output)
        .map(|index| {
            let center = (f64::from(index) + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as u32;
            let end = ((center + support + 0.5) as i64).min(i64::from(input)) as u32;
            let weights: Vec<f64> = (start..end)
                .map(|x| cubic((f64::from(x) - center + 0.5) * inverse))
                .collect();
            let sum: f64 = weights.iter().sum();
            let weights = weights
                .into_iter()
                .map(|weight| {
                    let normalized = if sum != 0.0 { weight / sum } else { weight };
                    let fixed = normalized * f64::from(1 << PRECISION);
                    if normalized < 0.0 {
                        (fixed - 0.5) as i32
                    } else {
                        (fixed + 0.5) as i32
                    }
                })
                .collect();
            Kernel { start, weights }
        })
        .collect()
}

pub(crate) fn bicubic(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let horizontal = if image.width() == width {
        image.clone()
    } else {
        let kernels = coefficients(image.width(), width);
        let mut output = RgbImage::new(width, image.height());
        for (input_row, output_row) in image
            .as_raw()
            .chunks_exact(image.width() as usize * 3)
            .zip(output.as_mut().chunks_exact_mut(width as usize * 3))
        {
            for (pixel, kernel) in output_row.chunks_exact_mut(3).zip(&kernels) {
                let start = kernel.start as usize * 3;
                let samples = &input_row[start..start + kernel.weights.len() * 3];
                let mut sums = [1_i64 << (PRECISION - 1); 3];
                for (rgb, &weight) in samples.chunks_exact(3).zip(&kernel.weights) {
                    for channel in 0..3 {
                        sums[channel] += i64::from(rgb[channel]) * i64::from(weight);
                    }
                }
                for (channel, sum) in pixel.iter_mut().zip(sums) {
                    *channel = (sum >> PRECISION).clamp(0, 255) as u8;
                }
            }
        }
        output
    };
    if image.height() == height {
        return horizontal;
    }
    let kernels = coefficients(image.height(), height);
    let row_bytes = width as usize * 3;
    let mut output = RgbImage::new(width, height);
    let mut sums = vec![0_i64; row_bytes];
    for (output_row, kernel) in output.as_mut().chunks_exact_mut(row_bytes).zip(&kernels) {
        sums.fill(1_i64 << (PRECISION - 1));
        let start = kernel.start as usize * row_bytes;
        let samples = &horizontal.as_raw()[start..start + kernel.weights.len() * row_bytes];
        for (input_row, &weight) in samples.chunks_exact(row_bytes).zip(&kernel.weights) {
            for (sum, &value) in sums.iter_mut().zip(input_row) {
                *sum += i64::from(value) * i64::from(weight);
            }
        }
        for (value, &sum) in output_row.iter_mut().zip(&sums) {
            *value = (sum >> PRECISION).clamp(0, 255) as u8;
        }
    }
    output
}

#[cfg(test)]
pub(crate) fn reference_bicubic(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    use image::Rgb;
    fn sample(values: impl Iterator<Item = (Rgb<u8>, i32)>) -> Rgb<u8> {
        let mut sums = [1_i64 << (PRECISION - 1); 3];
        for (pixel, weight) in values {
            for channel in 0..3 {
                sums[channel] += i64::from(pixel[channel]) * i64::from(weight);
            }
        }
        Rgb(sums.map(|sum| (sum >> PRECISION).clamp(0, 255) as u8))
    }
    let horizontal = if image.width() == width {
        image.clone()
    } else {
        let kernels = coefficients(image.width(), width);
        RgbImage::from_fn(width, image.height(), |x, y| {
            let kernel = &kernels[x as usize];
            sample(kernel.weights.iter().enumerate().map(|(offset, weight)| {
                (*image.get_pixel(kernel.start + offset as u32, y), *weight)
            }))
        })
    };
    if image.height() == height {
        return horizontal;
    }
    let kernels = coefficients(image.height(), height);
    RgbImage::from_fn(width, height, |x, y| {
        let kernel = &kernels[y as usize];
        sample(kernel.weights.iter().enumerate().map(|(offset, weight)| {
            (
                *horizontal.get_pixel(x, kernel.start + offset as u32),
                *weight,
            )
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn constant_color_survives_both_resize_directions() {
        let image = RgbImage::from_pixel(17, 9, Rgb([15, 123, 254]));
        for (width, height) in [(2, 3), (23, 21), (17, 9)] {
            assert!(
                bicubic(&image, width, height)
                    .pixels()
                    .all(|p| p.0 == [15, 123, 254])
            );
        }
    }

    #[test]
    fn mixed_axis_resize_matches_pillow_12_3_golden_bytes() {
        let image = RgbImage::from_raw(
            3,
            2,
            vec![
                0, 100, 255, 255, 0, 0, 10, 255, 50, 255, 255, 255, 0, 0, 0, 200, 10, 150,
            ],
        )
        .unwrap();
        // Pillow Image.resize((2, 4), Image.Resampling.BICUBIC), RGB, 2026-09-20.
        assert_eq!(
            bicubic(&image, 2, 4).into_raw(),
            vec![
                94, 45, 164, 105, 176, 16, 111, 78, 163, 109, 128, 35, 146, 142, 161, 117, 34, 74,
                163, 175, 160, 121, 0, 93
            ]
        );
    }
}
