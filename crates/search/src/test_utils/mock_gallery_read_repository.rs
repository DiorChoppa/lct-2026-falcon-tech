use std::collections::HashMap;
use std::sync::Mutex;

use crate::application::repositories::{CandidateRow, GalleryReadRepository};
use crate::domain::Tag;

#[derive(Default)]
pub struct MockGalleryReadRepository {
    pub rows: Vec<CandidateRow>,
    pub tags: HashMap<i64, Vec<Tag>>,
    pub received_knn_calls: Mutex<Vec<(Vec<f32>, String, u32)>>,
}

impl MockGalleryReadRepository {
    pub fn with_rows(rows: Vec<CandidateRow>) -> Self {
        Self {
            rows,
            ..Self::default()
        }
    }

    pub fn received_knn_calls(&self) -> Vec<(Vec<f32>, String, u32)> {
        self.received_knn_calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl GalleryReadRepository for MockGalleryReadRepository {
    async fn knn(
        &self,
        embedding: &[f32],
        model_version: &str,
        top_n: u32,
    ) -> anyhow::Result<Vec<CandidateRow>> {
        self.received_knn_calls.lock().unwrap().push((
            embedding.to_vec(),
            model_version.to_string(),
            top_n,
        ));
        Ok(self.rows.iter().take(top_n as usize).cloned().collect())
    }

    async fn get_tags(&self, gallery_id: i64) -> anyhow::Result<Vec<Tag>> {
        Ok(self.tags.get(&gallery_id).cloned().unwrap_or_default())
    }

    async fn get_crop_uri(&self, gallery_id: i64) -> anyhow::Result<Option<String>> {
        Ok(self
            .rows
            .iter()
            .find(|r| r.gallery_id == gallery_id)
            .map(|r| r.crop_uri.clone()))
    }
}
