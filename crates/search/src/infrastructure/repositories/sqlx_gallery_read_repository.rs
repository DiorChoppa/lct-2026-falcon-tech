use common::BBox;
use sqlx::PgPool;

use crate::application::repositories::{CandidateRow, GalleryReadRepository};
use crate::domain::Tag;

/// Connects as `search_ro`: SELECT-only on `gallery_items`/`tags`, enforced
/// by the role's grants (crates/gallery/migrations/0001_init.sql), not just
/// by this code.
pub struct SqlxGalleryReadRepository {
    pool: PgPool,
}

impl SqlxGalleryReadRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[derive(sqlx::FromRow)]
struct KnnRow {
    id: i64,
    crop_uri: String,
    plate: Option<String>,
    score: f32,
}

#[derive(sqlx::FromRow)]
struct TagRow {
    tag: String,
    confidence: f32,
    box_x: Option<i32>,
    box_y: Option<i32>,
    box_w: Option<i32>,
    box_h: Option<i32>,
}

impl From<TagRow> for Tag {
    fn from(r: TagRow) -> Self {
        let region = match (r.box_x, r.box_y, r.box_w, r.box_h) {
            (Some(x), Some(y), Some(w), Some(h)) => Some(BBox {
                x: x as u32,
                y: y as u32,
                w: w as u32,
                h: h as u32,
            }),
            _ => None,
        };
        Tag {
            key: r.tag,
            confidence: r.confidence,
            region,
        }
    }
}

#[async_trait::async_trait]
impl GalleryReadRepository for SqlxGalleryReadRepository {
    async fn knn(
        &self,
        embedding: &[f32],
        model_version: &str,
        top_n: u32,
    ) -> anyhow::Result<Vec<CandidateRow>> {
        // 1 - cosine_distance: <=> is pgvector's cosine distance operator,
        // and the HNSW index (crates/gallery/migrations) is built for it.
        // `<=>` yields float8; cast to float4 so sqlx can decode into `f32`
        // (found on the first live search, 21.09).
        let rows: Vec<KnnRow> = sqlx::query_as(
            "SELECT id, crop_uri, plate, (1 - (embedding <=> $1))::float4 AS score \
             FROM gallery_items WHERE model_version = $2 \
             ORDER BY embedding <=> $1 LIMIT $3",
        )
        .bind(pgvector::Vector::from(embedding.to_vec()))
        .bind(model_version)
        .bind(top_n as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| CandidateRow {
                gallery_id: r.id,
                score: r.score,
                crop_uri: r.crop_uri,
                plate: r.plate,
            })
            .collect())
    }

    async fn get_tags(&self, gallery_id: i64) -> anyhow::Result<Vec<Tag>> {
        let rows: Vec<TagRow> = sqlx::query_as(
            "SELECT tag, confidence, box_x, box_y, box_w, box_h FROM tags WHERE item_id = $1",
        )
        .bind(gallery_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    async fn get_crop_uri(&self, gallery_id: i64) -> anyhow::Result<Option<String>> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT crop_uri FROM gallery_items WHERE id = $1")
                .bind(gallery_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(|(uri,)| uri))
    }
}
