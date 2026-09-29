use chrono::{DateTime, Utc};
use common::BBox;
use sqlx::types::Json;
use sqlx::PgPool;

use crate::application::repositories::SearchRepository;
use crate::domain::{Candidate, NewSearch, SearchRecord};

pub struct SqlxSearchRepository {
    pool: PgPool,
}

impl SqlxSearchRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[derive(sqlx::FromRow)]
struct SearchRow {
    id: i64,
    query_crop_uri: String,
    bbox_x: i32,
    bbox_y: i32,
    bbox_w: i32,
    bbox_h: i32,
    result: Json<Vec<Candidate>>,
    accepted: bool,
    created_at: DateTime<Utc>,
}

impl From<SearchRow> for SearchRecord {
    fn from(r: SearchRow) -> Self {
        SearchRecord {
            id: r.id,
            query_crop_uri: r.query_crop_uri,
            bbox: BBox {
                x: r.bbox_x as u32,
                y: r.bbox_y as u32,
                w: r.bbox_w as u32,
                h: r.bbox_h as u32,
            },
            candidates: r.result.0,
            accepted: r.accepted,
            created_at: r.created_at,
        }
    }
}

const COLUMNS: &str =
    "id, query_crop_uri, bbox_x, bbox_y, bbox_w, bbox_h, result, accepted, created_at";

#[async_trait::async_trait]
impl SearchRepository for SqlxSearchRepository {
    async fn insert(&self, new: NewSearch) -> anyhow::Result<SearchRecord> {
        let row: SearchRow = sqlx::query_as(&format!(
            "INSERT INTO searches (query_crop_uri, bbox_x, bbox_y, bbox_w, bbox_h, result, accepted) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING {COLUMNS}"
        ))
        .bind(&new.query_crop_uri)
        .bind(new.bbox.x as i32)
        .bind(new.bbox.y as i32)
        .bind(new.bbox.w as i32)
        .bind(new.bbox.h as i32)
        .bind(Json(&new.candidates))
        .bind(new.accepted)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.into())
    }

    async fn get(&self, id: i64) -> anyhow::Result<Option<SearchRecord>> {
        let row: Option<SearchRow> =
            sqlx::query_as(&format!("SELECT {COLUMNS} FROM searches WHERE id = $1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(Into::into))
    }
}
