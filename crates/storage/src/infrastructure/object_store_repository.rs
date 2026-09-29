use std::path::Path;
use std::time::Duration;

use url::Url;

use crate::application::repositories::ObjectRepository;

/// `object_store`-backed bucket, scoped to `base` (e.g. "s3://gallery-crops"
/// or "file:///data/crops/gallery-crops"). `parse_url` dispatches on scheme;
/// schemes without their feature compiled in (e.g. `s3://` without the `aws`
/// feature) fail with a clear error rather than silently doing the wrong
/// thing.
pub struct ObjectStoreRepository {
    base: Url,
}

impl ObjectStoreRepository {
    pub fn new(base_url: &str) -> anyhow::Result<Self> {
        let base = Url::parse(base_url)
            .map_err(|err| anyhow::anyhow!("bucket url '{base_url}': {err}"))?;
        Ok(Self { base })
    }

    fn full_uri(&self, key: &str) -> String {
        format!("{}/{key}", self.base.as_str().trim_end_matches('/'))
    }

    /// `Url` normalizes ".." before `object_store` sees the path, so this
    /// checks the already-normalized `url.path()`/host against `base` rather
    /// than rejecting literal ".." in the input string.
    fn check_scope(&self, url: &Url) -> anyhow::Result<()> {
        anyhow::ensure!(
            url.scheme() == self.base.scheme() && url.host_str() == self.base.host_str(),
            "uri '{url}' is not in bucket {}",
            self.base
        );
        if url.scheme() == "file" {
            let requested = Path::new(url.path());
            let root = Path::new(self.base.path());
            anyhow::ensure!(
                requested.starts_with(root),
                "uri '{url}' is outside bucket root {}",
                self.base
            );
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl ObjectRepository for ObjectStoreRepository {
    async fn get(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        let url = Url::parse(uri).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;
        self.check_scope(&url)?;

        let (store, path) =
            object_store::parse_url(&url).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;
        let bytes = store
            .get(&path)
            .await
            .map_err(|err| anyhow::anyhow!("reading {uri}: {err}"))?
            .bytes()
            .await
            .map_err(|err| anyhow::anyhow!("reading {uri}: {err}"))?;
        Ok(bytes.to_vec())
    }

    async fn put(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<String> {
        let uri = self.full_uri(key);
        let url = Url::parse(&uri).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;

        let (store, path) =
            object_store::parse_url(&url).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;
        store
            .put(&path, bytes.into())
            .await
            .map_err(|err| anyhow::anyhow!("writing {uri}: {err}"))?;
        Ok(uri)
    }

    async fn delete(&self, uri: &str) -> anyhow::Result<()> {
        let url = Url::parse(uri).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;
        self.check_scope(&url)?;

        let (store, path) =
            object_store::parse_url(&url).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;
        store
            .delete(&path)
            .await
            .map_err(|err| anyhow::anyhow!("deleting {uri}: {err}"))
    }

    async fn presign(&self, uri: &str, _ttl: Duration) -> anyhow::Result<String> {
        let url = Url::parse(uri).map_err(|err| anyhow::anyhow!("uri '{uri}': {err}"))?;
        anyhow::bail!(
            "presigned URLs not supported for scheme '{}' yet; serve via /files/... instead",
            url.scheme()
        )
    }
}
