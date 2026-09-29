//! Галерея: шлюз только валидирует вход и переводит ответы gallery в DTO.
//! Кроп, эмбеддинг и запись делает gallery; кропы браузер получает через
//! `cropUrl` (см. crops.rs).

use std::io::Cursor;
use std::path::{Path, PathBuf};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::Json;
use chrono::{DateTime, Utc};
use common::BBox;
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

use crate::crops::crop_url;
use crate::error::ApiError;
use crate::extract::{Form, Id, JsonBody};
use crate::AppState;

/// Кадр не больше 3 МиБ: серверы gallery/search/inference принимают сообщения
/// до 4 МБ (дефолт tonic), а полные кадры 1080p весят сотни КБ каждый. Пачка
/// gallery.Import — по 64 кадра (соглашение proto), но не больше тех же 3 МиБ.
pub(crate) const MAX_FRAME_BYTES: usize = 3 << 20;
const IMPORT_BATCH: usize = 64;

/// proto i32 → пиксели без отрицательных.
fn clamp_bbox(x: i32, y: i32, w: i32, h: i32) -> Bbox {
    Bbox {
        x: x.max(0) as u32,
        y: y.max(0) as u32,
        w: w.max(0) as u32,
        h: h.max(0) as u32,
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Copy)]
pub struct Bbox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl From<Bbox> for BBox {
    fn from(b: Bbox) -> Self {
        BBox {
            x: b.x,
            y: b.y,
            w: b.w,
            h: b.h,
        }
    }
}

impl From<Option<proto::BBox>> for Bbox {
    fn from(b: Option<proto::BBox>) -> Self {
        let b = b.unwrap_or_default();
        clamp_bbox(b.x, b.y, b.w, b.h)
    }
}

impl From<proto::Region> for Bbox {
    fn from(r: proto::Region) -> Self {
        clamp_bbox(r.x, r.y, r.w, r.h)
    }
}

pub(crate) fn to_proto_bbox(b: BBox) -> proto::BBox {
    proto::BBox {
        x: b.x as i32,
        y: b.y as i32,
        w: b.w as i32,
        h: b.h as i32,
    }
}

/// Тег детали от tagger (через gallery.SetTags).
#[derive(Serialize, ToSchema)]
pub struct Tag {
    pub key: String,
    pub confidence: f32,
    /// Область на кропе в пикселях кропа; null, если тег без локализации.
    pub region: Option<Bbox>,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct GalleryItem {
    pub id: i64,
    /// Имя кадра без расширения (из датасета) или сгенерированный id.
    pub image_id: String,
    pub vehicle_id: Option<String>,
    pub bbox: Bbox,
    /// Кроп ТС: `/files/crops/...` через шлюз либо URL хранилища.
    pub crop_url: String,
    /// ГРЗ, привязанный оператором.
    pub plate: Option<String>,
    pub tags: Vec<Tag>,
    pub created_at: DateTime<Utc>,
}

fn non_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

impl GalleryItem {
    pub fn from_proto(p: proto::GalleryItem, crops_dir: &Path) -> Self {
        GalleryItem {
            id: p.id,
            image_id: p.image_id,
            vehicle_id: non_empty(p.vehicle_id),
            bbox: p.bbox.into(),
            crop_url: crop_url(&p.crop_uri, crops_dir),
            plate: non_empty(p.plate),
            tags: p
                .tags
                .into_iter()
                .map(|t| Tag {
                    key: t.key,
                    confidence: t.confidence,
                    region: t.region.map(Into::into),
                })
                .collect(),
            created_at: DateTime::from_timestamp(p.created_at_unix, 0).unwrap_or_default(),
        }
    }
}

/// Размеры кадра по заголовку файла, без полного декодирования: кадр 1080p
/// декодирует уже gallery/search/inference, шлюзу нужны только w×h для bbox.
pub(crate) fn frame_dimensions(bytes: &[u8]) -> Result<(u32, u32), ApiError> {
    let unrecognized = |e: &dyn std::fmt::Display| {
        ApiError::validation(format!("изображение не распознано (JPEG/PNG): {e}"))
    };
    image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| unrecognized(&e))?
        .into_dimensions()
        .map_err(|e| unrecognized(&e))
}

/// Общая валидация форм с кадром: поля есть, картинка распознана, bbox в кадре.
pub(crate) fn validate_frame(
    image: Option<Vec<u8>>,
    bbox: Option<String>,
) -> Result<(Vec<u8>, BBox), ApiError> {
    let image = image.ok_or_else(|| ApiError::validation("нет поля image"))?;
    if image.len() > MAX_FRAME_BYTES {
        return Err(ApiError::validation(format!(
            "кадр больше {} МиБ",
            MAX_FRAME_BYTES >> 20
        )));
    }
    let bbox: Bbox =
        serde_json::from_str(&bbox.ok_or_else(|| ApiError::validation("нет поля bbox"))?)
            .map_err(|e| ApiError::validation(format!("bbox: {e}")))?;
    let bbox: BBox = bbox.into();
    let (w, h) = frame_dimensions(&image)?;
    bbox.validate(w, h)
        .map_err(|e| ApiError::validation(e.to_string()))?;
    Ok((image, bbox))
}

/// Поля формы POST /api/gallery (multipart/form-data).
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct CreateForm {
    /// Полный кадр JPEG/PNG, до 3 МиБ.
    #[schema(value_type = String, format = Binary)]
    image: Vec<u8>,
    /// JSON `{"x":..,"y":..,"w":..,"h":..}` в пикселях кадра.
    bbox: String,
    image_id: Option<String>,
    vehicle_id: Option<String>,
    plate: Option<String>,
}

/// Добавить ТС в галерею: кадр + bbox → gallery.Add (кроп, эмбеддинг, запись).
#[utoipa::path(post, path = "/api/gallery", tag = "gallery",
    request_body(content = CreateForm, content_type = "multipart/form-data"),
    responses(
        (status = 201, body = GalleryItem),
        (status = 422, body = crate::error::ErrorBody, description = "bbox вне кадра, нет изображения"),
        (status = 502, body = crate::error::ErrorBody, description = "gallery или inference недоступен")))]
pub async fn create(
    State(s): State<AppState>,
    Form(mut form): Form,
) -> Result<(StatusCode, Json<GalleryItem>), ApiError> {
    let (mut image, mut bbox, mut image_id, mut vehicle_id, mut plate) =
        (None, None, None, None, None);
    while let Some(field) = form.next_field().await? {
        match field.name().unwrap_or_default() {
            "image" => image = Some(field.bytes().await?.to_vec()),
            "bbox" => bbox = Some(field.text().await?),
            "image_id" => image_id = Some(field.text().await?),
            "vehicle_id" => vehicle_id = Some(field.text().await?),
            "plate" => plate = Some(field.text().await?),
            _ => {}
        }
    }
    let (image, bbox) = validate_frame(image, bbox)?;
    // Пустые строки gallery трактует как «нет»: image_id генерирует, остальное — NULL.
    let item = s
        .gallery
        .add(proto::AddRequest {
            image,
            bbox: Some(to_proto_bbox(bbox)),
            image_id: image_id.unwrap_or_default(),
            vehicle_id: vehicle_id.unwrap_or_default(),
            plate: plate.unwrap_or_default(),
        })
        .await
        .map_err(ApiError::gallery)?;
    Ok((
        StatusCode::CREATED,
        Json(GalleryItem::from_proto(item, &s.crops_dir)),
    ))
}

/// Строка CSV датасета: test_*.csv без vehicle_id/camera_id, train.csv с ними.
#[derive(Deserialize)]
struct CsvRow {
    image_id: String,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    #[serde(default)]
    vehicle_id: Option<String>,
}

/// Поля формы POST /api/gallery/import.
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ImportForm {
    /// CSV в формате датасета: image_id,x,y,w,h[,vehicle_id,camera_id].
    #[schema(value_type = String, format = Binary)]
    csv: Vec<u8>,
    /// Каталог с кадрами `<image_id>.jpg` относительно DATASET_DIR сервиса. По умолчанию `images`.
    images_dir: Option<String>,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ImportError {
    pub image_id: String,
    pub message: String,
}

#[derive(Serialize, ToSchema)]
pub struct ImportReport {
    pub imported: usize,
    pub failed: usize,
    pub errors: Vec<ImportError>,
}

fn images_dir(dataset_dir: &Path, rel: Option<String>) -> Result<PathBuf, ApiError> {
    let rel = rel.unwrap_or_else(|| "images".into());
    if rel.contains("..") || rel.starts_with('/') {
        return Err(ApiError::validation(
            "images_dir: только каталог внутри DATASET_DIR",
        ));
    }
    Ok(dataset_dir.join(rel))
}

/// Массовый импорт кадров датасета по CSV: шлюз читает кадры с диска и шлёт
/// пачки в gallery.Import. bbox обрезается по кадру (в разметке встречаются
/// выходы на 1–2 px). Битые кадры не прерывают импорт, недоступность gallery
/// или inference — прерывает (уже импортированные записи остаются).
#[utoipa::path(post, path = "/api/gallery/import", tag = "gallery",
    request_body(content = ImportForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, body = ImportReport),
        (status = 422, body = crate::error::ErrorBody),
        (status = 502, body = crate::error::ErrorBody, description = "gallery или inference недоступен")))]
pub async fn import(
    State(s): State<AppState>,
    Form(mut form): Form,
) -> Result<Json<ImportReport>, ApiError> {
    let (mut csv, mut rel) = (None, None);
    while let Some(field) = form.next_field().await? {
        match field.name().unwrap_or_default() {
            "csv" => csv = Some(field.bytes().await?.to_vec()),
            "images_dir" => rel = Some(field.text().await?),
            _ => {}
        }
    }
    let csv = csv.ok_or_else(|| ApiError::validation("нет поля csv"))?;
    let dir = images_dir(&s.dataset_dir, rel)?;
    let rows: Vec<CsvRow> = csv::Reader::from_reader(csv.as_slice())
        .deserialize()
        .collect::<Result<_, _>>()
        .map_err(|e| ApiError::validation(format!("csv: {e}")))?;

    let mut report = ImportReport {
        imported: 0,
        failed: 0,
        errors: vec![],
    };
    let mut batch: Vec<proto::ImportItem> = Vec::new();
    let mut batch_bytes = 0;
    for row in &rows {
        let (bytes, w, h) = match load_frame(&dir, &row.image_id).await {
            Ok(v) => v,
            Err(e) => {
                report.failed += 1;
                report.errors.push(ImportError {
                    image_id: row.image_id.clone(),
                    message: e.message,
                });
                continue;
            }
        };
        if !batch.is_empty()
            && (batch.len() >= IMPORT_BATCH || batch_bytes + bytes.len() > MAX_FRAME_BYTES)
        {
            flush(&s, &mut batch, &mut report).await?;
            batch_bytes = 0;
        }
        let bbox = BBox {
            x: row.x,
            y: row.y,
            w: row.w,
            h: row.h,
        }
        .clamp(w, h);
        batch_bytes += bytes.len();
        batch.push(proto::ImportItem {
            image: bytes,
            bbox: Some(to_proto_bbox(bbox)),
            image_id: row.image_id.clone(),
            vehicle_id: row.vehicle_id.clone().unwrap_or_default(),
        });
    }
    if !batch.is_empty() {
        flush(&s, &mut batch, &mut report).await?;
    }
    Ok(Json(report))
}

/// Кадр `<dir>/<image_id>.jpg` из датасета: image_id — только имя файла (он же
/// ключ объекта в хранилище gallery), кадр — распознанная картинка не больше
/// MAX_FRAME_BYTES.
async fn load_frame(dir: &Path, image_id: &str) -> Result<(Vec<u8>, u32, u32), ApiError> {
    if image_id.is_empty() || image_id.contains(['/', '\\']) || image_id.contains("..") {
        return Err(ApiError::validation("image_id: недопустимое имя"));
    }
    let path = dir.join(format!("{image_id}.jpg"));
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|e| ApiError::validation(format!("{}: {e}", path.display())))?;
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(ApiError::validation(format!(
            "кадр больше {} МиБ",
            MAX_FRAME_BYTES >> 20
        )));
    }
    let (w, h) = frame_dimensions(&bytes)?;
    Ok((bytes, w, h))
}

/// Одна пачка в gallery.Import; поэлементные ошибки gallery — в отчёт.
async fn flush(
    s: &AppState,
    batch: &mut Vec<proto::ImportItem>,
    report: &mut ImportReport,
) -> Result<(), ApiError> {
    let resp = s
        .gallery
        .import(std::mem::take(batch))
        .await
        .map_err(ApiError::gallery)?;
    report.imported += resp.imported.max(0) as usize;
    report.failed += resp.failed.max(0) as usize;
    report
        .errors
        .extend(resp.errors.into_iter().map(|e| ImportError {
            image_id: e.image_id,
            message: e.message,
        }));
    Ok(())
}

#[derive(Deserialize, IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct PageQuery {
    /// С 1.
    pub page: Option<u32>,
    /// 1..200, по умолчанию 20.
    pub page_size: Option<u32>,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Pagination {
    pub page: u32,
    pub page_size: u32,
    pub total_items: u64,
    pub total_pages: u32,
}

#[derive(Serialize, ToSchema)]
pub struct GalleryPage {
    pub data: Vec<GalleryItem>,
    pub pagination: Pagination,
}

/// Список галереи, новые записи первыми.
#[utoipa::path(get, path = "/api/gallery", tag = "gallery", params(PageQuery),
    responses((status = 200, body = GalleryPage), (status = 502, body = crate::error::ErrorBody)))]
pub async fn list(
    State(s): State<AppState>,
    Query(q): Query<PageQuery>,
) -> Result<Json<GalleryPage>, ApiError> {
    let page = q.page.unwrap_or(1).max(1);
    let page_size = q.page_size.unwrap_or(20).clamp(1, 200);
    let resp = s
        .gallery
        .list(page, page_size)
        .await
        .map_err(ApiError::gallery)?;
    Ok(Json(GalleryPage {
        data: resp
            .items
            .into_iter()
            .map(|p| GalleryItem::from_proto(p, &s.crops_dir))
            .collect(),
        pagination: Pagination {
            page,
            page_size,
            total_items: resp.total_items,
            total_pages: resp.total_items.div_ceil(page_size as u64) as u32,
        },
    }))
}

#[derive(Deserialize, ToSchema)]
pub struct PlateBody {
    /// ГРЗ; null — снять привязку.
    pub plate: Option<String>,
}

/// Привязать ГРЗ к записи (из ALPR другой камеры или вручную) либо снять его.
#[utoipa::path(put, path = "/api/gallery/{id}/plate", tag = "gallery",
    params(("id" = i64, Path, description = "id записи")),
    request_body = PlateBody,
    responses(
        (status = 200, body = GalleryItem),
        (status = 404, body = crate::error::ErrorBody),
        (status = 422, body = crate::error::ErrorBody)))]
pub async fn set_plate(
    State(s): State<AppState>,
    Id(id): Id,
    JsonBody(body): JsonBody<PlateBody>,
) -> Result<Json<GalleryItem>, ApiError> {
    let plate = match body.plate.map(|p| p.trim().to_string()) {
        Some(p) if p.is_empty() => {
            return Err(ApiError::validation(
                "plate: пустая строка; чтобы снять ГРЗ, передайте null",
            ))
        }
        Some(p) if p.chars().count() > 32 => {
            return Err(ApiError::validation("plate: длиннее 32 символов"))
        }
        other => other,
    };
    let item = s
        .gallery
        .set_plate(id, plate.unwrap_or_default())
        .await
        .map_err(ApiError::gallery)?;
    Ok(Json(GalleryItem::from_proto(item, &s.crops_dir)))
}

/// Одна запись галереи.
#[utoipa::path(get, path = "/api/gallery/{id}", tag = "gallery",
    params(("id" = i64, Path, description = "id записи")),
    responses((status = 200, body = GalleryItem), (status = 404, body = crate::error::ErrorBody)))]
pub async fn get_one(State(s): State<AppState>, Id(id): Id) -> Result<Json<GalleryItem>, ApiError> {
    let item = s.gallery.get(id).await.map_err(ApiError::gallery)?;
    Ok(Json(GalleryItem::from_proto(item, &s.crops_dir)))
}
