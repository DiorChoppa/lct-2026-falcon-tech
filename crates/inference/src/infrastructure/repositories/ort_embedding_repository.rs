use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::application::repositories::{EmbedOutput, EmbeddingRepository};
use crate::domain::ModelManifest;

/// ONNX Runtime, CPU execution provider (default, no CUDA/CoreML EP
/// registered). The session loads lazily on the first `embed()` call rather
/// than in the constructor: until real weights exist (`models/model.json` is
/// currently a stub manifest), the service must still start and answer
/// health/Info — only the `Embed` call itself fails, with a clear error.
pub struct OrtEmbeddingRepository {
    model_path: PathBuf,
    input_name: String,
    output_name: String,
    patches_output_name: Option<String>,
    session: Mutex<Option<Session>>,
}

impl OrtEmbeddingRepository {
    /// `model_dir` is the directory holding the manifest; the model file is
    /// resolved from `manifest.file` relative to it.
    pub fn new(manifest: &ModelManifest, model_dir: &Path) -> Self {
        Self {
            model_path: model_dir.join(&manifest.file),
            input_name: manifest.input_name.clone(),
            output_name: manifest.output_name.clone(),
            patches_output_name: manifest.patches_output_name.clone(),
            session: Mutex::new(None),
        }
    }
}

/// Splits a flattened ORT output tensor into one chunk per batch element.
fn split_by_batch(shape: &[i64], data: &[f32], n: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    anyhow::ensure!(
        shape[0] as usize == n,
        "model returned batch size {}, expected {}",
        shape[0],
        n
    );
    let per_element = if n == 0 { 0 } else { data.len() / n };
    Ok(data
        .chunks_exact(per_element)
        .map(|chunk| chunk.to_vec())
        .collect())
}

impl EmbeddingRepository for OrtEmbeddingRepository {
    fn embed(&self, batch: Array4<f32>, with_patches: bool) -> anyhow::Result<Vec<EmbedOutput>> {
        let n = batch.shape()[0];

        if with_patches {
            anyhow::ensure!(
                self.patches_output_name.is_some(),
                "model has no patches output configured"
            );
        }

        let mut guard = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("ORT session mutex poisoned"))?;
        if guard.is_none() {
            let session = Session::builder()
                .map_err(|err| anyhow::anyhow!("{err}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|err| anyhow::anyhow!("{err}"))?
                .commit_from_file(&self.model_path)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "failed to load ONNX model at {}: {err}",
                        self.model_path.display()
                    )
                })?;
            *guard = Some(session);
        }
        let session = guard.as_mut().expect("session initialized above");

        let tensor = Tensor::from_array(batch).map_err(|err| anyhow::anyhow!("{err}"))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|err| anyhow::anyhow!("{err}"))?;

        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let embeddings = split_by_batch(shape, data, n)?;

        let mut patches: Vec<Option<Vec<f32>>> = vec![None; embeddings.len()];
        if with_patches {
            // Checked above: with_patches=true implies patches_output_name is Some.
            let patches_output_name = self.patches_output_name.as_deref().expect("checked above");
            let (shape, data) = outputs[patches_output_name]
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow::anyhow!("{err}"))?;
            let chunks = split_by_batch(shape, data, n)?;
            patches = chunks.into_iter().map(Some).collect();
        }

        Ok(embeddings
            .into_iter()
            .zip(patches)
            .map(|(embedding, patches)| EmbedOutput { embedding, patches })
            .collect())
    }
}
