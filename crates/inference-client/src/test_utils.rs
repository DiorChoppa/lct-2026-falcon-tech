use std::sync::Mutex;

use crate::{Embedded, EmbeddingClient, Frame, ModelInfo};

pub struct MockEmbeddingClient {
    /// Returned by `embed()`.
    pub embedded: Vec<Embedded>,
    /// Returned by `embed_by_uri()`; defaults to `embedded` if not set via
    /// `with_embedded_by_uri` — most tests only exercise one of the two.
    pub embedded_by_uri: Vec<Embedded>,
    pub info: ModelInfo,
    pub received_frame_counts: Mutex<Vec<usize>>,
    pub received_uris: Mutex<Vec<Vec<String>>>,
    pub received_with_patches: Mutex<Vec<bool>>,
}

impl Default for MockEmbeddingClient {
    fn default() -> Self {
        let embedded = vec![Embedded {
            embedding: vec![1.0, 0.0, 0.0],
            patches: None,
        }];
        Self {
            embedded_by_uri: embedded.clone(),
            embedded,
            info: ModelInfo {
                model_version: "v1".into(),
                dim: 3,
                input_height: 4,
                input_width: 4,
                supports_patches: false,
                patch_grid_h: 0,
                patch_grid_w: 0,
                patch_dim: 0,
            },
            received_frame_counts: Mutex::new(vec![]),
            received_uris: Mutex::new(vec![]),
            received_with_patches: Mutex::new(vec![]),
        }
    }
}

impl MockEmbeddingClient {
    pub fn with_embedded(embedded: Vec<Embedded>) -> Self {
        Self {
            embedded_by_uri: embedded.clone(),
            embedded,
            ..Self::default()
        }
    }

    pub fn with_embedded_by_uri(mut self, embedded_by_uri: Vec<Embedded>) -> Self {
        self.embedded_by_uri = embedded_by_uri;
        self
    }

    pub fn with_info(info: ModelInfo) -> Self {
        Self {
            info,
            ..Self::default()
        }
    }

    pub fn received_frame_counts(&self) -> Vec<usize> {
        self.received_frame_counts.lock().unwrap().clone()
    }

    pub fn received_uris(&self) -> Vec<Vec<String>> {
        self.received_uris.lock().unwrap().clone()
    }

    pub fn received_with_patches(&self) -> Vec<bool> {
        self.received_with_patches.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for MockEmbeddingClient {
    async fn embed(&self, frames: &[Frame], with_patches: bool) -> anyhow::Result<Vec<Embedded>> {
        self.received_frame_counts
            .lock()
            .unwrap()
            .push(frames.len());
        self.received_with_patches
            .lock()
            .unwrap()
            .push(with_patches);
        Ok(self.embedded.clone())
    }

    async fn embed_by_uri(
        &self,
        uris: &[String],
        with_patches: bool,
    ) -> anyhow::Result<Vec<Embedded>> {
        self.received_uris.lock().unwrap().push(uris.to_vec());
        self.received_with_patches
            .lock()
            .unwrap()
            .push(with_patches);
        Ok(self.embedded_by_uri.clone())
    }

    async fn info(&self) -> anyhow::Result<ModelInfo> {
        Ok(self.info.clone())
    }
}
