use std::sync::Mutex;

use chrono::Utc;

use crate::application::repositories::SearchRepository;
use crate::domain::{NewSearch, SearchRecord};

#[derive(Default)]
pub struct MockSearchRepository {
    records: Mutex<Vec<SearchRecord>>,
    next_id: Mutex<i64>,
}

#[async_trait::async_trait]
impl SearchRepository for MockSearchRepository {
    async fn insert(&self, new: NewSearch) -> anyhow::Result<SearchRecord> {
        let mut next_id = self.next_id.lock().unwrap();
        *next_id += 1;
        let record = SearchRecord {
            id: *next_id,
            query_crop_uri: new.query_crop_uri,
            bbox: new.bbox,
            candidates: new.candidates,
            accepted: new.accepted,
            created_at: Utc::now(),
        };
        self.records.lock().unwrap().push(record.clone());
        Ok(record)
    }

    async fn get(&self, id: i64) -> anyhow::Result<Option<SearchRecord>> {
        Ok(self
            .records
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .cloned())
    }
}
