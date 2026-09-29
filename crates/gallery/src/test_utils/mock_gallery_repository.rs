use std::sync::Mutex;

use chrono::Utc;

use crate::application::repositories::GalleryRepository;
use crate::domain::{GalleryItem, NewItem, Tag};

/// In-memory fake, not just a recorder: use_cases tests exercise real
/// insert/get/list/delete round trips without a live Postgres.
#[derive(Default)]
pub struct MockGalleryRepository {
    items: Mutex<Vec<GalleryItem>>,
    next_id: Mutex<i64>,
}

#[async_trait::async_trait]
impl GalleryRepository for MockGalleryRepository {
    async fn insert(&self, item: NewItem) -> anyhow::Result<GalleryItem> {
        let mut next_id = self.next_id.lock().unwrap();
        *next_id += 1;
        let item = GalleryItem {
            id: *next_id,
            image_id: item.image_id,
            vehicle_id: item.vehicle_id,
            bbox: item.bbox,
            crop_uri: item.crop_uri,
            plate: item.plate,
            tags: vec![],
            model_version: item.model_version,
            created_at: Utc::now(),
        };
        self.items.lock().unwrap().push(item.clone());
        Ok(item)
    }

    async fn get(&self, id: i64) -> anyhow::Result<Option<GalleryItem>> {
        Ok(self
            .items
            .lock()
            .unwrap()
            .iter()
            .find(|i| i.id == id)
            .cloned())
    }

    async fn list(&self, page: u32, page_size: u32) -> anyhow::Result<(Vec<GalleryItem>, u64)> {
        let items = self.items.lock().unwrap();
        let total = items.len() as u64;
        let start = ((page - 1) as usize) * page_size as usize;
        let page_items = items
            .iter()
            .rev()
            .skip(start)
            .take(page_size as usize)
            .cloned()
            .collect();
        Ok((page_items, total))
    }

    async fn set_plate(
        &self,
        id: i64,
        plate: Option<String>,
    ) -> anyhow::Result<Option<GalleryItem>> {
        let mut items = self.items.lock().unwrap();
        let Some(item) = items.iter_mut().find(|i| i.id == id) else {
            return Ok(None);
        };
        item.plate = plate;
        Ok(Some(item.clone()))
    }

    async fn delete(&self, id: i64) -> anyhow::Result<Option<GalleryItem>> {
        let mut items = self.items.lock().unwrap();
        let Some(pos) = items.iter().position(|i| i.id == id) else {
            return Ok(None);
        };
        Ok(Some(items.remove(pos)))
    }

    async fn set_tags(&self, item_id: i64, tags: Vec<Tag>) -> anyhow::Result<()> {
        let mut items = self.items.lock().unwrap();
        if let Some(item) = items.iter_mut().find(|i| i.id == item_id) {
            item.tags = tags;
        }
        Ok(())
    }

    async fn stale_items(
        &self,
        current_version: &str,
        only_version: Option<&str>,
    ) -> anyhow::Result<Vec<GalleryItem>> {
        let items = self.items.lock().unwrap();
        Ok(items
            .iter()
            .filter(|i| match only_version {
                Some(v) => i.model_version == v,
                None => i.model_version != current_version,
            })
            .cloned()
            .collect())
    }

    async fn update_embedding(
        &self,
        id: i64,
        _embedding: Vec<f32>,
        model_version: &str,
    ) -> anyhow::Result<()> {
        let mut items = self.items.lock().unwrap();
        if let Some(item) = items.iter_mut().find(|i| i.id == id) {
            item.model_version = model_version.to_string();
        }
        Ok(())
    }
}
