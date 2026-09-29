use std::path::PathBuf;

use url::Url;

/// Resolves `CropRef.uri` (an `object_store` URL — `file://` locally,
/// `s3://` once `crates/storage` lands) and loads the bytes.
pub struct CropLoader {
    file_root: PathBuf,
}

impl CropLoader {
    pub fn new(file_root: PathBuf) -> Self {
        Self { file_root }
    }

    pub async fn load(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        let url = Url::parse(uri).map_err(|err| anyhow::anyhow!("crop_uri '{uri}': {err}"))?;
        self.check_file_scope(&url)?;

        let (store, path) = object_store::parse_url(&url)
            .map_err(|err| anyhow::anyhow!("crop_uri '{uri}': {err}"))?;
        let bytes = store
            .get(&path)
            .await
            .map_err(|err| anyhow::anyhow!("reading crop {uri}: {err}"))?
            .bytes()
            .await
            .map_err(|err| anyhow::anyhow!("reading crop {uri}: {err}"))?;
        Ok(bytes.to_vec())
    }

    /// `Url` normalizes ".." before `object_store` sees the path, so this
    /// checks the decoded local file path against the allow-listed
    /// root rather than rejecting literal ".." in the input string.
    fn check_file_scope(&self, url: &Url) -> anyhow::Result<()> {
        if url.scheme() != "file" {
            return Ok(());
        }
        // Component-wise: a string-prefix check would let root "/data/crops"
        // match "/data/crops-other/...".
        let requested = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid local file crop_uri: {url}"))?;
        anyhow::ensure!(
            requested.starts_with(&self.file_root),
            "file:// crop_uri must be under {}, got {}",
            self.file_root.display(),
            url.path()
        );
        Ok(())
    }
}
