use proto::tagger_client::TaggerClient as ProtoTaggerClient;
use proto::EnqueueRequest;

use crate::application::repositories::TaggerClient;

pub struct GrpcTaggerClient {
    pub url: String,
}

#[async_trait::async_trait]
impl TaggerClient for GrpcTaggerClient {
    async fn enqueue(&self, item_id: i64, crop: Vec<u8>) -> anyhow::Result<()> {
        let mut client = ProtoTaggerClient::connect(self.url.clone())
            .await
            .map_err(|err| anyhow::anyhow!("tagger unavailable at {}: {err}", self.url))?;
        client
            .enqueue(EnqueueRequest { item_id, crop })
            .await
            .map_err(|status| anyhow::anyhow!("tagger Enqueue failed: {status}"))?;
        Ok(())
    }
}
