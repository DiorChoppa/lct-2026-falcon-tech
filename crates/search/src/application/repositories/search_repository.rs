use crate::domain::{NewSearch, SearchRecord};

#[async_trait::async_trait]
pub trait SearchRepository: Send + Sync {
    async fn insert(&self, new: NewSearch) -> anyhow::Result<SearchRecord>;
    async fn get(&self, id: i64) -> anyhow::Result<Option<SearchRecord>>;
}
