mod helpers;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use helpers::{l2_normalize, preprocess_batch};

use crate::application::repositories::EmbeddingRepository;
use crate::application::structures::ModelInfo;
use crate::domain::{Crop, ModelManifest};

/// One crop's result: embedding always present, patch tokens only when
/// `with_patches` was requested (see `UseCases::embed`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbedResult {
    pub embedding: Vec<f32>,
    pub patches: Option<Vec<f32>>,
}

pub struct UseCases {
    manifest: ModelManifest,
    embedding_repository: Arc<dyn EmbeddingRepository + Send + Sync>,
}

impl UseCases {
    pub fn new(
        manifest: ModelManifest,
        embedding_repository: Arc<dyn EmbeddingRepository + Send + Sync>,
    ) -> Self {
        Self {
            manifest,
            embedding_repository,
        }
    }

    /// Decode → crop → resize → normalize → `EmbeddingRepository` → L2-normalize
    /// unless the model already does it (`manifest.l2_normalized`).
    ///
    /// `with_patches` additionally requests patch tokens; fails if the loaded
    /// model has no patches output (check `info().supports_patches` first).
    pub fn embed(&self, crops: &[Crop], with_patches: bool) -> anyhow::Result<Vec<EmbedResult>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        if with_patches {
            anyhow::ensure!(
                self.manifest.supports_patches(),
                "model {} has no patches output; check Info.supports_patches first",
                self.manifest.name
            );
        }

        let batch = preprocess_batch(crops, &self.manifest)?;
        let outputs = self.embedding_repository.embed(batch, with_patches)?;

        Ok(outputs
            .into_iter()
            .map(|mut output| {
                if !self.manifest.l2_normalized {
                    l2_normalize(&mut output.embedding);
                    if let (Some(patches), Some(patch_dim)) =
                        (output.patches.as_mut(), self.manifest.patch_dim)
                    {
                        for chunk in patches.chunks_mut(patch_dim as usize) {
                            l2_normalize(chunk);
                        }
                    }
                }
                EmbedResult {
                    embedding: output.embedding,
                    patches: output.patches,
                }
            })
            .collect())
    }

    pub fn info(&self) -> ModelInfo {
        ModelInfo::from(&self.manifest)
    }
}
