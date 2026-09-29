use std::path::Path;

use url::Url;

use crate::infrastructure::grpc::crop_loader::CropLoader;

fn file_url(path: &Path) -> Url {
    Url::from_file_path(path).unwrap()
}

#[tokio::test]
async fn load_reads_file_under_configured_root() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("crop.bin"), b"hello").unwrap();
    let loader = CropLoader::new(dir.path().to_path_buf());

    let bytes = loader
        .load(file_url(&dir.path().join("crop.bin")).as_str())
        .await
        .unwrap();

    assert_eq!(b"hello".to_vec(), bytes);
}

#[tokio::test]
async fn load_rejects_file_uri_escaping_root_via_dot_dot() {
    // Two sibling tempdirs (both direct children of std::env::temp_dir())
    // so "<allowed>/../<secret_dir_name>/secret.txt" deterministically
    // escapes `allowed` by exactly one level, regardless of nesting depth.
    let allowed = tempfile::tempdir().unwrap();
    let secret_dir = tempfile::tempdir().unwrap();
    std::fs::write(secret_dir.path().join("secret.txt"), b"secret").unwrap();
    let loader = CropLoader::new(allowed.path().to_path_buf());

    let escaping = format!(
        "{}/../{}/secret.txt",
        file_url(allowed.path()).as_str(),
        secret_dir.path().file_name().unwrap().to_str().unwrap()
    );

    let err = loader.load(&escaping).await.unwrap_err();

    assert!(err.to_string().contains("must be under"));
}

#[tokio::test]
async fn load_rejects_file_uri_outside_root() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    let loader = CropLoader::new(dir.path().to_path_buf());

    let err = loader
        .load(file_url(outside.path()).as_str())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("must be under"));
}

#[tokio::test]
async fn load_rejects_sibling_dir_with_matching_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let sibling = format!("{}-other", dir.path().display());
    std::fs::create_dir_all(&sibling).unwrap();
    std::fs::write(format!("{sibling}/x.jpg"), b"data").unwrap();
    let loader = CropLoader::new(dir.path().to_path_buf());

    let escaping_path = Path::new(&sibling).join("x.jpg");
    let err = loader
        .load(file_url(&escaping_path).as_str())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("must be under"));
    std::fs::remove_dir_all(&sibling).unwrap();
}

#[tokio::test]
async fn load_rejects_malformed_uri() {
    let loader = CropLoader::new(Path::new("/data/crops").to_path_buf());

    let err = loader.load("not-a-url").await.unwrap_err();

    assert!(err.to_string().contains("crop_uri"));
}

#[tokio::test]
async fn load_rejects_unconfigured_scheme() {
    let loader = CropLoader::new(Path::new("/data/crops").to_path_buf());

    let err = loader.load("s3://bucket/key.jpg").await.unwrap_err();

    assert!(err.to_string().contains("crop_uri"));
}

#[tokio::test]
async fn load_fails_for_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let loader = CropLoader::new(dir.path().to_path_buf());

    let err = loader
        .load(file_url(&dir.path().join("missing.bin")).as_str())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("reading crop"));
}
