use std::time::Duration;

/// One bucket's worth of crop storage (`gallery-crops` or `query-crops`),
/// each service holding its own scoped instance per 04-architecture.md §5.
#[async_trait::async_trait]
pub trait ObjectRepository: Send + Sync {
    async fn get(&self, uri: &str) -> anyhow::Result<Vec<u8>>;

    /// Returns the crop_uri the bytes were written to.
    async fn put(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<String>;

    async fn delete(&self, uri: &str) -> anyhow::Result<()>;

    async fn presign(&self, uri: &str, ttl: Duration) -> anyhow::Result<String>;
}
