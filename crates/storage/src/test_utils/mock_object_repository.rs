use std::sync::Mutex;
use std::time::Duration;

use crate::application::repositories::ObjectRepository;

pub struct MockObjectRepository {
    pub bytes: Vec<u8>,
    pub put_prefix: String,
    pub received_puts: Mutex<Vec<(String, Vec<u8>)>>,
    pub received_gets: Mutex<Vec<String>>,
    pub received_deletes: Mutex<Vec<String>>,
}

impl Default for MockObjectRepository {
    fn default() -> Self {
        Self {
            bytes: vec![],
            put_prefix: "mock://bucket".into(),
            received_puts: Mutex::new(vec![]),
            received_gets: Mutex::new(vec![]),
            received_deletes: Mutex::new(vec![]),
        }
    }
}

impl MockObjectRepository {
    pub fn with_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ..Self::default()
        }
    }

    pub fn received_puts(&self) -> Vec<(String, Vec<u8>)> {
        self.received_puts.lock().unwrap().clone()
    }

    pub fn received_gets(&self) -> Vec<String> {
        self.received_gets.lock().unwrap().clone()
    }

    pub fn received_deletes(&self) -> Vec<String> {
        self.received_deletes.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ObjectRepository for MockObjectRepository {
    async fn get(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        self.received_gets.lock().unwrap().push(uri.to_string());
        Ok(self.bytes.clone())
    }

    async fn put(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<String> {
        let uri = format!("{}/{key}", self.put_prefix);
        self.received_puts
            .lock()
            .unwrap()
            .push((key.to_string(), bytes));
        Ok(uri)
    }

    async fn delete(&self, uri: &str) -> anyhow::Result<()> {
        self.received_deletes.lock().unwrap().push(uri.to_string());
        Ok(())
    }

    async fn presign(&self, uri: &str, _ttl: Duration) -> anyhow::Result<String> {
        Ok(format!("{uri}?presigned=mock"))
    }
}
