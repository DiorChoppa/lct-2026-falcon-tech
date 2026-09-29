//! `crop_uri` из gallery/search → URL картинки для браузера. Пока хранилище
//! `file://` под общим томом CROPS_DIR, шлюз отдаёт файлы сам через
//! `/files/crops/...`; для `s3://` URI presigned URL появится позже — сейчас
//! URI возвращается как есть.

use std::path::Path;

pub fn crop_url(uri: &str, crops_dir: &Path) -> String {
    let Some(path) = uri.strip_prefix("file://") else {
        return uri.to_string();
    };
    let root = crops_dir.to_string_lossy();
    let root = root.trim_end_matches('/');
    match path
        .strip_prefix(root)
        .and_then(|rest| rest.strip_prefix('/'))
    {
        Some(rest) if !root.is_empty() => format!("/files/crops/{rest}"),
        _ => {
            tracing::warn!(
                uri,
                crops_dir = root,
                "crop_uri вне CROPS_DIR, отдаём как есть"
            );
            uri.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_uri_under_crops_dir_becomes_files_url() {
        assert_eq!(
            crop_url(
                "file:///app/data/crops/gallery-crops/abc.jpg",
                Path::new("/app/data/crops")
            ),
            "/files/crops/gallery-crops/abc.jpg"
        );
        assert_eq!(
            crop_url(
                "file:///app/data/crops/query-crops/q.jpg",
                Path::new("/app/data/crops/")
            ),
            "/files/crops/query-crops/q.jpg"
        );
    }

    #[test]
    fn file_uri_outside_crops_dir_is_kept() {
        assert_eq!(
            crop_url("file:///tmp/other/x.jpg", Path::new("/app/data/crops")),
            "file:///tmp/other/x.jpg"
        );
        // Префикс совпадает по строке, но это другой каталог.
        assert_eq!(
            crop_url(
                "file:///app/data/crops2/x.jpg",
                Path::new("/app/data/crops")
            ),
            "file:///app/data/crops2/x.jpg"
        );
    }

    #[test]
    fn s3_uri_is_kept() {
        assert_eq!(
            crop_url("s3://gallery-crops/1.jpg", Path::new("/app/data/crops")),
            "s3://gallery-crops/1.jpg"
        );
    }
}
