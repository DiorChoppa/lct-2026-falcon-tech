mod grpc_client;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

pub use grpc_client::GrpcEmbeddingClient;

use common::BBox;

/// One frame and the bboxes to embed in it — matches gRPC Embed's shape.
pub struct Frame {
    pub image: Vec<u8>,
    pub boxes: Vec<BBox>,
}

/// One bbox's result: embedding always present, patch tokens only when
/// `with_patches` was requested and the model supports them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Embedded {
    pub embedding: Vec<f32>,
    pub patches: Option<Vec<f32>>,
}

/// Loaded model metadata — mirrors inference's `Info` RPC.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    pub model_version: String,
    pub dim: u32,
    pub input_height: u32,
    pub input_width: u32,
    pub supports_patches: bool,
    pub patch_grid_h: u32,
    pub patch_grid_w: u32,
    pub patch_dim: u32,
}

/// gallery's and search's only dependency on inference — depending on this
/// crate instead of `crates/inference` directly avoids pulling in `ort` for
/// services that never run the model themselves.
#[async_trait::async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, frames: &[Frame], with_patches: bool) -> anyhow::Result<Vec<Embedded>>;

    /// Embeds already-stored crops by URI (`EmbedRequest.crops`) — inference
    /// reads the bytes itself, so search never needs read access to
    /// gallery-crops just to rerank. Results are in `uris` order.
    async fn embed_by_uri(
        &self,
        uris: &[String],
        with_patches: bool,
    ) -> anyhow::Result<Vec<Embedded>>;

    async fn info(&self) -> anyhow::Result<ModelInfo>;
}
