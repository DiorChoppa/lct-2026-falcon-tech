use ndarray::Array4;

/// One crop's raw model output: embedding always present, patch tokens only
/// when `with_patches` was requested and the model exposes them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbedOutput {
    pub embedding: Vec<f32>,
    pub patches: Option<Vec<f32>>,
}

/// Runs an already-preprocessed NCHW batch through the model, one `EmbedOutput`
/// per element (no L2-normalization — that's the use case's job, see
/// `l2_normalized`).
pub trait EmbeddingRepository {
    fn embed(&self, batch: Array4<f32>, with_patches: bool) -> anyhow::Result<Vec<EmbedOutput>>;
}
