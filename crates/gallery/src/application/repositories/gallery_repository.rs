use crate::domain::{GalleryItem, NewItem, Tag};

#[async_trait::async_trait]
pub trait GalleryRepository: Send + Sync {
    async fn insert(&self, item: NewItem) -> anyhow::Result<GalleryItem>;
    async fn get(&self, id: i64) -> anyhow::Result<Option<GalleryItem>>;
    /// Returns the page and the total row count (for pagination metadata).
    async fn list(&self, page: u32, page_size: u32) -> anyhow::Result<(Vec<GalleryItem>, u64)>;
    async fn set_plate(
        &self,
        id: i64,
        plate: Option<String>,
    ) -> anyhow::Result<Option<GalleryItem>>;
    /// Returns the deleted item (so the use case can also remove its crop
    /// object), or `None` if `id` didn't exist.
    async fn delete(&self, id: i64) -> anyhow::Result<Option<GalleryItem>>;
    async fn set_tags(&self, item_id: i64, tags: Vec<Tag>) -> anyhow::Result<()>;
    /// `only_version = Some(v)` restricts to items on exactly `v`; `None`
    /// means every item not already on `current_version`.
    async fn stale_items(
        &self,
        current_version: &str,
        only_version: Option<&str>,
    ) -> anyhow::Result<Vec<GalleryItem>>;
    async fn update_embedding(
        &self,
        id: i64,
        embedding: Vec<f32>,
        model_version: &str,
    ) -> anyhow::Result<()>;
}
