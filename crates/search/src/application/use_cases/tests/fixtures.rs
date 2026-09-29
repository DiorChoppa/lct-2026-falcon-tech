use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};

pub fn solid_png() -> Vec<u8> {
    let img = RgbImage::from_pixel(4, 4, image::Rgb([80, 40, 160]));
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&img, 4, 4, ExtendedColorType::Rgb8)
        .unwrap();
    out
}
