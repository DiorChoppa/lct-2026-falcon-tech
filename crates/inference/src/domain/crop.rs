use std::sync::Arc;

use common::BBox;

/// A single bbox on a source frame: the whole frame (JPEG/PNG) plus a rectangle.
/// `image` is an `Arc` so multiple `Crop`s from the same frame don't clone the bytes.
#[derive(Debug, Clone)]
pub struct Crop {
    pub image: Arc<[u8]>,
    pub bbox: BBox,
}
