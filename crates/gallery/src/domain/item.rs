use chrono::{DateTime, Utc};
use common::BBox;

use super::Tag;

#[derive(Debug, Clone, PartialEq)]
pub struct GalleryItem {
    pub id: i64,
    pub image_id: String,
    pub vehicle_id: Option<String>,
    pub bbox: BBox,
    pub crop_uri: String,
    pub plate: Option<String>,
    pub tags: Vec<Tag>,
    pub model_version: String,
    pub created_at: DateTime<Utc>,
}

/// A record not yet assigned an id or `created_at` — what `GalleryRepository::insert` takes.
#[derive(Debug, Clone, PartialEq)]
pub struct NewItem {
    pub image_id: String,
    pub vehicle_id: Option<String>,
    pub bbox: BBox,
    pub crop_uri: String,
    pub embedding: Vec<f32>,
    pub model_version: String,
    pub plate: Option<String>,
}
