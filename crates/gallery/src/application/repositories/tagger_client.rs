/// Fire-and-forget: gallery never waits on tagging, and a down tagger must
/// never fail Add/Import (04-architecture.md §6 — tagger failure loses only
/// in-flight tags, `Reindex --tags` catches up later).
#[async_trait::async_trait]
pub trait TaggerClient: Send + Sync {
    async fn enqueue(&self, item_id: i64, crop: Vec<u8>) -> anyhow::Result<()>;
}
