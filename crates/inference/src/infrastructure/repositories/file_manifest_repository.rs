use std::path::PathBuf;

use crate::application::repositories::ManifestRepository;
use crate::domain::ModelManifest;

pub struct FileManifestRepository {
    path: PathBuf,
}

impl FileManifestRepository {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl ManifestRepository for FileManifestRepository {
    fn load(&self) -> anyhow::Result<ModelManifest> {
        ModelManifest::load(&self.path)
    }
}
