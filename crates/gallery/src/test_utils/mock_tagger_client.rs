use std::sync::Mutex;

use crate::application::repositories::TaggerClient;

#[derive(Default)]
pub struct MockTaggerClient {
    pub received_enqueues: Mutex<Vec<(i64, Vec<u8>)>>,
}

impl MockTaggerClient {
    pub fn received_enqueues(&self) -> Vec<(i64, Vec<u8>)> {
        self.received_enqueues.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl TaggerClient for MockTaggerClient {
    async fn enqueue(&self, item_id: i64, crop: Vec<u8>) -> anyhow::Result<()> {
        self.received_enqueues.lock().unwrap().push((item_id, crop));
        Ok(())
    }
}
