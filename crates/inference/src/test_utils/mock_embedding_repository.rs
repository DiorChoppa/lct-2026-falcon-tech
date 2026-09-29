use std::sync::Mutex;

use ndarray::Array4;

use crate::application::repositories::{EmbedOutput, EmbeddingRepository};

pub struct MockEmbeddingRepository {
    pub outputs: Vec<EmbedOutput>,
    pub received_batches: Mutex<Vec<Array4<f32>>>,
    pub received_with_patches: Mutex<Vec<bool>>,
}

impl Default for MockEmbeddingRepository {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl MockEmbeddingRepository {
    /// Embedding-only outputs, no patches — the common case in tests.
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        Self::with_outputs(
            vectors
                .into_iter()
                .map(|embedding| EmbedOutput {
                    embedding,
                    patches: None,
                })
                .collect(),
        )
    }

    pub fn with_outputs(outputs: Vec<EmbedOutput>) -> Self {
        Self {
            outputs,
            received_batches: Mutex::new(vec![]),
            received_with_patches: Mutex::new(vec![]),
        }
    }

    pub fn received_batch_count(&self) -> usize {
        self.received_batches.lock().unwrap().len()
    }

    pub fn received_batch_shapes(&self) -> Vec<Vec<usize>> {
        self.received_batches
            .lock()
            .unwrap()
            .iter()
            .map(|batch| batch.shape().to_vec())
            .collect()
    }

    pub fn received_with_patches(&self) -> Vec<bool> {
        self.received_with_patches.lock().unwrap().clone()
    }
}

impl EmbeddingRepository for MockEmbeddingRepository {
    fn embed(&self, batch: Array4<f32>, with_patches: bool) -> anyhow::Result<Vec<EmbedOutput>> {
        self.received_batches.lock().unwrap().push(batch);
        self.received_with_patches
            .lock()
            .unwrap()
            .push(with_patches);
        Ok(self.outputs.clone())
    }
}
