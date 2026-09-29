use common::BBox;

/// Zero-shot detail tag from tagger, delivered via `Gallery.SetTags`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    pub key: String,
    pub confidence: f32,
    pub region: Option<BBox>,
}
