use crate::domain::ModelManifest;

/// Application-layer DTO: loaded model metadata, returned by the
/// GetModelInfo use case and mapped by infrastructure into proto::InfoResponse.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_name: String,
    pub model_version: String,
    pub dim: u32,
    pub input_height: u32,
    pub input_width: u32,
    pub execution_provider: String,
    pub supports_patches: bool,
    pub patch_grid_h: u32,
    pub patch_grid_w: u32,
    pub patch_dim: u32,
}

impl From<&ModelManifest> for ModelInfo {
    fn from(manifest: &ModelManifest) -> Self {
        ModelInfo {
            model_name: manifest.name.clone(),
            model_version: manifest.version.clone(),
            dim: manifest.dim,
            input_height: manifest.input_height,
            input_width: manifest.input_width,
            // TODO: report the actual execution provider once ort supports switching (cpu/cuda).
            execution_provider: "cpu".to_string(),
            supports_patches: manifest.supports_patches(),
            patch_grid_h: manifest.patch_grid_h.unwrap_or(0),
            patch_grid_w: manifest.patch_grid_w.unwrap_or(0),
            patch_dim: manifest.patch_dim.unwrap_or(0),
        }
    }
}
