use std::time::Duration;

use url::Url;

use crate::application::repositories::ObjectRepository;
use crate::infrastructure::ObjectStoreRepository;

fn bucket_url(dir: &std::path::Path) -> String {
    Url::from_file_path(dir).unwrap().to_string()
}

#[tokio::test]
async fn put_then_get_round_trips_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(dir.path())).unwrap();

    let uri = repo.put("abc.jpg", b"hello".to_vec()).await.unwrap();
    let bytes = repo.get(&uri).await.unwrap();

    assert_eq!(b"hello".to_vec(), bytes);
    assert!(uri.ends_with("/abc.jpg"));
}

#[tokio::test]
async fn get_rejects_uri_outside_bucket_root() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(dir.path())).unwrap();

    let err = repo
        .get(Url::from_file_path(outside.path()).unwrap().as_ref())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("outside bucket root"));
}

#[tokio::test]
async fn get_rejects_uri_escaping_root_via_dot_dot() {
    let allowed = tempfile::tempdir().unwrap();
    let secret_dir = tempfile::tempdir().unwrap();
    std::fs::write(secret_dir.path().join("secret.txt"), b"secret").unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(allowed.path())).unwrap();

    let escaping = format!(
        "{}/../{}/secret.txt",
        bucket_url(allowed.path()),
        secret_dir.path().file_name().unwrap().to_str().unwrap()
    );

    let err = repo.get(&escaping).await.unwrap_err();

    assert!(err.to_string().contains("outside bucket root"));
}

#[tokio::test]
async fn get_rejects_uri_from_a_sibling_directory() {
    // file:// URIs have no host, so this is caught by the root check, not the
    // scheme+host check (which matters once s3:// distinguishes buckets by host).
    let bucket_a = tempfile::tempdir().unwrap();
    let bucket_b = tempfile::tempdir().unwrap();
    std::fs::write(bucket_b.path().join("x.jpg"), b"data").unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(bucket_a.path())).unwrap();

    let foreign_uri = format!("{}/x.jpg", bucket_url(bucket_b.path()));
    let err = repo.get(&foreign_uri).await.unwrap_err();

    assert!(err.to_string().contains("outside bucket root"));
}

#[tokio::test]
async fn delete_removes_the_object() {
    let dir = tempfile::tempdir().unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(dir.path())).unwrap();
    let uri = repo.put("abc.jpg", b"hello".to_vec()).await.unwrap();

    repo.delete(&uri).await.unwrap();
    let err = repo.get(&uri).await.unwrap_err();

    assert!(err.to_string().contains("reading"));
}

#[tokio::test]
async fn delete_rejects_uri_outside_bucket_root() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(dir.path())).unwrap();

    let err = repo
        .delete(Url::from_file_path(outside.path()).unwrap().as_ref())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("outside bucket root"));
}

#[tokio::test]
async fn presign_fails_clearly_for_file_scheme() {
    let dir = tempfile::tempdir().unwrap();
    let repo = ObjectStoreRepository::new(&bucket_url(dir.path())).unwrap();
    let uri = repo.put("abc.jpg", b"hello".to_vec()).await.unwrap();

    let err = repo
        .presign(&uri, Duration::from_secs(900))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("not supported"));
}
