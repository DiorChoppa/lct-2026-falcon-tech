use chrono::{DateTime, Utc};
use common::BBox;
use serde::{Deserialize, Serialize};

/// Stored as JSONB in `searches.result` — see infrastructure::repositories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub gallery_id: i64,
    /// Raw global-vector cosine similarity, 0..1.
    pub score: f32,
    /// `score` after calibration and (if requested) patch rerank with
    /// `alpha` — the value `accepted` is decided on.
    pub confidence: f32,
    /// Only set when the caller requested patch rerank/details.
    pub local_score: Option<f32>,
    pub plate: Option<String>,
    pub crop_uri: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewSearch {
    pub query_crop_uri: String,
    pub bbox: BBox,
    pub candidates: Vec<Candidate>,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchRecord {
    pub id: i64,
    pub query_crop_uri: String,
    pub bbox: BBox,
    pub candidates: Vec<Candidate>,
    pub accepted: bool,
    pub created_at: DateTime<Utc>,
}
