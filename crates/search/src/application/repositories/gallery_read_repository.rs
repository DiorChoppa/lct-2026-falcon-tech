use crate::domain::Tag;

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRow {
    pub gallery_id: i64,
    /// Cosine similarity of the global vector, 0..1.
    pub score: f32,
    pub crop_uri: String,
    pub plate: Option<String>,
}

/// `search_ro`: SELECT-only on `gallery_items`/`tags`, never writes them —
/// that's `gallery`'s job.
#[async_trait::async_trait]
pub trait GalleryReadRepository: Send + Sync {
    async fn knn(
        &self,
        embedding: &[f32],
        model_version: &str,
        top_n: u32,
    ) -> anyhow::Result<Vec<CandidateRow>>;

    async fn get_tags(&self, gallery_id: i64) -> anyhow::Result<Vec<Tag>>;

    /// For `Compare`, which only has `gallery_id`, not the full row.
    async fn get_crop_uri(&self, gallery_id: i64) -> anyhow::Result<Option<String>>;
}
