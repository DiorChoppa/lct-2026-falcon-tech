use std::path::Path;

use serde::{Deserialize, Serialize};

/// ONNX model contract. models/model.json is written by the exporter in ml/
/// and read by inference and the parity test. Any change needs cross-team sign-off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    pub version: String,
    /// ONNX file name, relative to the manifest's directory.
    pub file: String,
    pub input_name: String,
    pub output_name: String,
    /// Input is NCHW float32, RGB.
    pub input_height: u32,
    pub input_width: u32,
    /// Normalization: (pixel/255 - mean) / std, per RGB channel.
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// "bilinear" | "bicubic" — must match what training used.
    pub resize: String,
    pub dim: u32,
    /// If false, inference normalizes the output itself.
    pub l2_normalized: bool,
    /// Second ONNX output holding patch tokens, for local (patch) matching in
    /// `search`. Absent for models that only export a global embedding.
    #[serde(default)]
    pub patches_output_name: Option<String>,
    #[serde(default)]
    pub patch_grid_h: Option<u32>,
    #[serde(default)]
    pub patch_grid_w: Option<u32>,
    /// Size of one patch token; the patches output flattens to
    /// [patch_grid_h * patch_grid_w * patch_dim] per crop.
    #[serde(default)]
    pub patch_dim: Option<u32>,
}

impl ModelManifest {
    /// Public crate API: used directly by reid-cli (no network) and by
    /// infrastructure::repositories::FileManifestRepository for the service.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn supports_patches(&self) -> bool {
        self.patches_output_name.is_some()
    }
}
