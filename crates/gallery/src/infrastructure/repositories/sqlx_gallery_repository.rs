use chrono::{DateTime, Utc};
use common::BBox;
use sqlx::PgPool;

use crate::application::repositories::GalleryRepository;
use crate::domain::{GalleryItem, NewItem, Tag};

pub struct SqlxGalleryRepository {
    pool: PgPool,
}

impl SqlxGalleryRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[derive(sqlx::FromRow)]
struct ItemRow {
    id: i64,
    image_id: String,
    vehicle_id: Option<String>,
    bbox_x: i32,
    bbox_y: i32,
    bbox_w: i32,
    bbox_h: i32,
    crop_uri: String,
    plate: Option<String>,
    model_version: String,
    created_at: DateTime<Utc>,
}

impl ItemRow {
    fn into_item(self, tags: Vec<Tag>) -> GalleryItem {
        GalleryItem {
            id: self.id,
            image_id: self.image_id,
            vehicle_id: self.vehicle_id,
            bbox: BBox {
                x: self.bbox_x as u32,
                y: self.bbox_y as u32,
                w: self.bbox_w as u32,
                h: self.bbox_h as u32,
            },
            crop_uri: self.crop_uri,
            plate: self.plate,
            tags,
            model_version: self.model_version,
            created_at: self.created_at,
        }
    }
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

const ITEM_COLUMNS: &str = "id, image_id, vehicle_id, bbox_x, bbox_y, bbox_w, bbox_h, crop_uri, \
                             plate, model_version, created_at";

impl SqlxGalleryRepository {
    async fn fetch_tags(&self, item_id: i64) -> anyhow::Result<Vec<Tag>> {
        let rows: Vec<TagRow> = sqlx::query_as(
            "SELECT tag, confidence, box_x, box_y, box_w, box_h FROM tags WHERE item_id = $1",
        )
        .bind(item_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(Into::into).collect())
    }
}

#[async_trait::async_trait]
impl GalleryRepository for SqlxGalleryRepository {
    async fn insert(&self, item: NewItem) -> anyhow::Result<GalleryItem> {
        let row: ItemRow = sqlx::query_as(&format!(
            "INSERT INTO gallery_items \
             (image_id, vehicle_id, bbox_x, bbox_y, bbox_w, bbox_h, crop_uri, embedding, \
              model_version, plate) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING {ITEM_COLUMNS}"
        ))
        .bind(&item.image_id)
        .bind(&item.vehicle_id)
        .bind(item.bbox.x as i32)
        .bind(item.bbox.y as i32)
        .bind(item.bbox.w as i32)
        .bind(item.bbox.h as i32)
        .bind(&item.crop_uri)
        .bind(pgvector::Vector::from(item.embedding))
        .bind(&item.model_version)
        .bind(&item.plate)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.into_item(vec![]))
    }

    async fn get(&self, id: i64) -> anyhow::Result<Option<GalleryItem>> {
        let row: Option<ItemRow> = sqlx::query_as(&format!(
            "SELECT {ITEM_COLUMNS} FROM gallery_items WHERE id = $1"
        ))
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else { return Ok(None) };
        let tags = self.fetch_tags(row.id).await?;
        Ok(Some(row.into_item(tags)))
    }

    async fn list(&self, page: u32, page_size: u32) -> anyhow::Result<(Vec<GalleryItem>, u64)> {
        let (total,): (i64,) = sqlx::query_as("SELECT count(*) FROM gallery_items")
            .fetch_one(&self.pool)
            .await?;
        let rows: Vec<ItemRow> = sqlx::query_as(&format!(
            "SELECT {ITEM_COLUMNS} FROM gallery_items ORDER BY id DESC LIMIT $1 OFFSET $2"
        ))
        .bind(page_size as i64)
        .bind(((page - 1) as i64) * page_size as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            let tags = self.fetch_tags(row.id).await?;
            items.push(row.into_item(tags));
        }
        Ok((items, total as u64))
    }

    async fn set_plate(
        &self,
        id: i64,
        plate: Option<String>,
    ) -> anyhow::Result<Option<GalleryItem>> {
        let row: Option<ItemRow> = sqlx::query_as(&format!(
            "UPDATE gallery_items SET plate = $2 WHERE id = $1 RETURNING {ITEM_COLUMNS}"
        ))
        .bind(id)
        .bind(&plate)
        .fetch_optional(&self.pool)
        .await?;
        match row {
            Some(row) => {
                let tags = self.fetch_tags(row.id).await?;
                Ok(Some(row.into_item(tags)))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: i64) -> anyhow::Result<Option<GalleryItem>> {
        // Tags cascade-delete with the row; the caller only needs crop_uri
        // from the returned item, so tags aren't fetched here.
        let row: Option<ItemRow> = sqlx::query_as(&format!(
            "DELETE FROM gallery_items WHERE id = $1 RETURNING {ITEM_COLUMNS}"
        ))
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| row.into_item(vec![])))
    }

    async fn set_tags(&self, item_id: i64, tags: Vec<Tag>) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM tags WHERE item_id = $1")
            .bind(item_id)
            .execute(&mut *tx)
            .await?;
        for tag in tags {
            let (box_x, box_y, box_w, box_h) = match tag.region {
                Some(r) => (
                    Some(r.x as i32),
                    Some(r.y as i32),
                    Some(r.w as i32),
                    Some(r.h as i32),
                ),
                None => (None, None, None, None),
            };
            sqlx::query(
                "INSERT INTO tags (item_id, tag, confidence, box_x, box_y, box_w, box_h) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7)",
            )
            .bind(item_id)
            .bind(tag.key)
            .bind(tag.confidence)
            .bind(box_x)
            .bind(box_y)
            .bind(box_w)
            .bind(box_h)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn stale_items(
        &self,
        current_version: &str,
        only_version: Option<&str>,
    ) -> anyhow::Result<Vec<GalleryItem>> {
        let rows: Vec<ItemRow> = match only_version {
            Some(v) => {
                sqlx::query_as(&format!(
                    "SELECT {ITEM_COLUMNS} FROM gallery_items WHERE model_version = $1"
                ))
                .bind(v)
                .fetch_all(&self.pool)
                .await?
            }
            None => {
                sqlx::query_as(&format!(
                    "SELECT {ITEM_COLUMNS} FROM gallery_items WHERE model_version <> $1"
                ))
                .bind(current_version)
                .fetch_all(&self.pool)
                .await?
            }
        };
        Ok(rows.into_iter().map(|row| row.into_item(vec![])).collect())
    }

    async fn update_embedding(
        &self,
        id: i64,
        embedding: Vec<f32>,
        model_version: &str,
    ) -> anyhow::Result<()> {
        sqlx::query("UPDATE gallery_items SET embedding = $2, model_version = $3 WHERE id = $1")
            .bind(id)
            .bind(pgvector::Vector::from(embedding))
            .bind(model_version)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
