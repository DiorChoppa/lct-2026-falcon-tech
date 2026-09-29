use common::BBox;

/// Read-only mirror of gallery's `Tag` — search only ever reads tags
/// (`search_ro` has SELECT on `tags`), never writes them.
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    pub key: String,
    pub confidence: f32,
    pub region: Option<BBox>,
}
