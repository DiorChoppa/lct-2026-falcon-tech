#[cfg(test)]
mod tests;

use std::sync::Arc;

use common::BBox;
use storage::ObjectRepository;

use crate::application::repositories::{EmbeddingClient, Frame, GalleryRepository, TaggerClient};
use crate::application::structures::{ImportItem, ImportReport, Page};
use crate::domain::{GalleryItem, NewItem, Tag};

/// crop_uri already points at a whole cropped image, so re-embedding it
/// (Reindex) treats the full image as the bbox — inference clamps this down
/// to the real dimensions once decoded. `i32::MAX`, not `u32::MAX`: proto::BBox
/// is int32 on the wire, and `u32::MAX as i32` is -1.
const FULL_IMAGE_BBOX: BBox = BBox {
    x: 0,
    y: 0,
    w: i32::MAX as u32,
    h: i32::MAX as u32,
};

const CROP_JPEG_QUALITY: u8 = 95;

pub struct UseCases {
    gallery_repository: Arc<dyn GalleryRepository>,
    object_repository: Arc<dyn ObjectRepository>,
    embedding_client: Arc<dyn EmbeddingClient>,
    tagger_client: Arc<dyn TaggerClient>,
}

impl UseCases {
    pub fn new(
        gallery_repository: Arc<dyn GalleryRepository>,
        object_repository: Arc<dyn ObjectRepository>,
        embedding_client: Arc<dyn EmbeddingClient>,
        tagger_client: Arc<dyn TaggerClient>,
    ) -> Self {
        Self {
            gallery_repository,
            object_repository,
            embedding_client,
            tagger_client,
        }
    }

    /// Best-effort, matching `delete`'s crop cleanup: a down tagger must
    /// never fail Add/Import, so failures are logged, not propagated.
    async fn enqueue_tagging(&self, item_id: i64, crop: Vec<u8>) {
        if let Err(err) = self.tagger_client.enqueue(item_id, crop).await {
            tracing::warn!(item_id, error = %err, "tagger.Enqueue failed; item has no tags for now");
        }
    }

    pub async fn add(
        &self,
        image: Vec<u8>,
        bbox: BBox,
        image_id: Option<String>,
        vehicle_id: Option<String>,
        plate: Option<String>,
    ) -> anyhow::Result<GalleryItem> {
        let image_id = non_empty(image_id).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let crop = common::crop_jpeg(&image, bbox, CROP_JPEG_QUALITY)?;
        let crop_uri = self
            .object_repository
            .put(&format!("{image_id}.jpg"), crop.clone())
            .await?;

        let embedding = self.embed_one(image, bbox).await?;
        let model_version = self.embedding_client.info().await?.model_version;

        let item = self
            .gallery_repository
            .insert(NewItem {
                image_id,
                vehicle_id: non_empty(vehicle_id),
                bbox,
                crop_uri,
                embedding,
                model_version,
                plate: non_empty(plate),
            })
            .await?;

        self.enqueue_tagging(item.id, crop).await;
        Ok(item)
    }

    /// Errors for individual items go to the report, not the `Result` — one
    /// bad frame in a batch shouldn't fail the rest (matches the dataset
    /// import contract: partial success is normal).
    pub async fn import(&self, items: Vec<ImportItem>) -> anyhow::Result<ImportReport> {
        let mut report = ImportReport::default();
        if items.is_empty() {
            return Ok(report);
        }

        let frames: Vec<Frame> = items
            .iter()
            .map(|item| Frame {
                image: item.image.clone(),
                boxes: vec![item.bbox],
            })
            .collect();
        let embedded = self.embedding_client.embed(&frames, false).await?;
        anyhow::ensure!(
            embedded.len() == items.len(),
            "expected {} embeddings, got {}",
            items.len(),
            embedded.len()
        );
        let model_version = self.embedding_client.info().await?.model_version;

        for (item, embedded) in items.into_iter().zip(embedded) {
            match self
                .import_one(item, embedded.embedding, model_version.clone())
                .await
            {
                Ok(_) => report.imported += 1,
                Err((image_id, message)) => {
                    report.failed += 1;
                    report.errors.push((image_id, message));
                }
            }
        }
        Ok(report)
    }

    async fn import_one(
        &self,
        item: ImportItem,
        embedding: Vec<f32>,
        model_version: String,
    ) -> Result<(), (String, String)> {
        let crop = common::crop_jpeg(&item.image, item.bbox, CROP_JPEG_QUALITY)
            .map_err(|err| (item.image_id.clone(), err.to_string()))?;
        let crop_uri = self
            .object_repository
            .put(&format!("{}.jpg", item.image_id), crop.clone())
            .await
            .map_err(|err| (item.image_id.clone(), err.to_string()))?;
        let inserted = self
            .gallery_repository
            .insert(NewItem {
                image_id: item.image_id.clone(),
                vehicle_id: item.vehicle_id,
                bbox: item.bbox,
                crop_uri,
                embedding,
                model_version,
                plate: None,
            })
            .await
            .map_err(|err| (item.image_id, err.to_string()))?;

        self.enqueue_tagging(inserted.id, crop).await;
        Ok(())
    }

    pub async fn get(&self, id: i64) -> anyhow::Result<Option<GalleryItem>> {
        self.gallery_repository.get(id).await
    }

    pub async fn list(&self, page: u32, page_size: u32) -> anyhow::Result<Page<GalleryItem>> {
        let page = page.max(1);
        let page_size = page_size.clamp(1, 200);
        let (items, total_items) = self.gallery_repository.list(page, page_size).await?;
        Ok(Page { items, total_items })
    }

    pub async fn set_plate(
        &self,
        id: i64,
        plate: Option<String>,
    ) -> anyhow::Result<Option<GalleryItem>> {
        let plate = match plate.map(|p| p.trim().to_string()) {
            Some(p) if p.is_empty() => {
                anyhow::bail!("plate must not be an empty string; omit it to clear the plate")
            }
            Some(p) if p.chars().count() > 32 => anyhow::bail!("plate longer than 32 characters"),
            other => other,
        };
        self.gallery_repository.set_plate(id, plate).await
    }

    /// Best-effort: the DB row is the source of truth, so a failed crop
    /// delete doesn't fail the whole operation — it just leaves an orphaned
    /// object for a future cleanup pass.
    pub async fn delete(&self, id: i64) -> anyhow::Result<bool> {
        let Some(item) = self.gallery_repository.delete(id).await? else {
            return Ok(false);
        };
        if let Err(err) = self.object_repository.delete(&item.crop_uri).await {
            tracing::warn!(id, crop_uri = %item.crop_uri, error = %err, "failed to delete crop object");
        }
        Ok(true)
    }

    pub async fn set_tags(&self, item_id: i64, tags: Vec<Tag>) -> anyhow::Result<()> {
        self.gallery_repository.set_tags(item_id, tags).await
    }

    pub async fn reindex(&self, only_version: Option<String>) -> anyhow::Result<u32> {
        let current = self.embedding_client.info().await?.model_version;
        let stale = self
            .gallery_repository
            .stale_items(&current, only_version.as_deref())
            .await?;

        let mut reindexed = 0;
        for item in stale {
            let bytes = self.object_repository.get(&item.crop_uri).await?;
            let embedding = self.embed_one(bytes, FULL_IMAGE_BBOX).await?;
            self.gallery_repository
                .update_embedding(item.id, embedding, &current)
                .await?;
            reindexed += 1;
        }
        Ok(reindexed)
    }

    async fn embed_one(&self, image: Vec<u8>, bbox: BBox) -> anyhow::Result<Vec<f32>> {
        let embedded = self
            .embedding_client
            .embed(
                &[Frame {
                    image,
                    boxes: vec![bbox],
                }],
                false,
            )
            .await?;
        embedded
            .into_iter()
            .next()
            .map(|e| e.embedding)
            .ok_or_else(|| anyhow::anyhow!("inference returned no embedding"))
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}
