use crate::domain::ModelManifest;

/// Model manifest loading abstraction for application use cases.
/// Implemented by infrastructure::repositories::FileManifestRepository.
pub trait ManifestRepository {
    fn load(&self) -> anyhow::Result<ModelManifest>;
}
