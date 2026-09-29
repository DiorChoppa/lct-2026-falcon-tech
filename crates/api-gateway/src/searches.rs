//! Поиск и история: search.Search решает о кандидатах и отказе (порог из
//! model.json применяет он), шлюз только дособирает карточки записей из
//! gallery и отдаёт CSV экспорта как есть.

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::DateTime;
use serde::Serialize;
use utoipa::ToSchema;

use crate::crops::crop_url;
use crate::error::ApiError;
use crate::extract::{CompareIds, Form, Id};
use crate::gallery::{to_proto_bbox, validate_frame, Bbox, GalleryItem, Tag};
use crate::AppState;

const DEFAULT_TOP_N: u32 = 10;
const MAX_TOP_N: u32 = 100;

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub item: GalleryItem,
    /// Косинусное сходство глобальных эмбеддингов, 0..1.
    pub score: f32,
    /// Уверенность, к которой search применяет `threshold`: `alpha*score +
    /// (1-alpha)*localScore`; при alpha = 1 равна score. Та же, что в export.csv.
    pub confidence: f32,
    /// Счёт по патчам, если search его считал (alpha < 1 в model.json).
    pub local_score: Option<f32>,
    /// Прошёл ли кандидат порог режима отказа.
    pub accepted: bool,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchResponse {
    /// id поиска: экспорт — /api/searches/{id}/export.csv.
    pub id: i64,
    pub query_crop_url: String,
    /// Порог из models/model.json (тот же, что применяет search).
    pub threshold: f32,
    /// Топ-N по убыванию уверенности, включая ниже порога. Ни одного accepted — отказ.
    pub candidates: Vec<Candidate>,
}

/// Поля формы POST /api/search (multipart/form-data).
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct SearchForm {
    /// Полный кадр JPEG/PNG, до 3 МиБ.
    #[schema(value_type = String, format = Binary)]
    image: Vec<u8>,
    /// JSON `{"x":..,"y":..,"w":..,"h":..}` в пикселях кадра.
    bbox: String,
    /// Сколько кандидатов вернуть, 1..100, по умолчанию 10.
    top_n: Option<u32>,
}

/// Карточка кандидата, когда gallery недоступен: остановка gallery не должна
/// ломать поиск (04-architecture.md §1), а search знает только id, ГРЗ и кроп.
fn partial_item(c: &proto::Candidate, s: &AppState) -> GalleryItem {
    GalleryItem {
        id: c.gallery_id,
        image_id: String::new(),
        vehicle_id: None,
        bbox: Bbox {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        crop_url: crop_url(&c.crop_uri, &s.crops_dir),
        plate: (!c.plate.is_empty()).then(|| c.plate.clone()),
        tags: vec![],
        created_at: DateTime::UNIX_EPOCH,
    }
}

/// Найти ТС в галерее: кадр + bbox → search.Search → кандидаты с карточками из gallery.
#[utoipa::path(post, path = "/api/search", tag = "search",
    request_body(content = SearchForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, body = SearchResponse),
        (status = 422, body = crate::error::ErrorBody, description = "bbox вне кадра, нет изображения"),
        (status = 502, body = crate::error::ErrorBody, description = "search или inference недоступен")))]
pub async fn search(
    State(s): State<AppState>,
    Form(mut form): Form,
) -> Result<Json<SearchResponse>, ApiError> {
    let (mut image, mut bbox, mut top_n) = (None, None, None);
    while let Some(field) = form.next_field().await? {
        match field.name().unwrap_or_default() {
            "image" => image = Some(field.bytes().await?.to_vec()),
            "bbox" => bbox = Some(field.text().await?),
            "top_n" => top_n = Some(field.text().await?),
            _ => {}
        }
    }
    let top_n = match top_n.filter(|v| !v.is_empty()) {
        None => DEFAULT_TOP_N,
        Some(v) => v
            .parse::<u32>()
            .map_err(|_| ApiError::validation("top_n: целое число"))?
            .clamp(1, MAX_TOP_N),
    };
    let (image, bbox) = validate_frame(image, bbox)?;

    let resp = s
        .search
        .search(proto::SearchRequest {
            image,
            bbox: Some(to_proto_bbox(bbox)),
            top_n,
            details: false,
        })
        .await
        .map_err(ApiError::search)?;

    // search отдаёт весь топ-N, отказ — флаг; карточки — параллельно из gallery.
    let items =
        futures::future::join_all(resp.candidates.iter().map(|c| s.gallery.get(c.gallery_id)))
            .await;
    let candidates = resp
        .candidates
        .iter()
        .zip(items)
        .map(|(c, item)| {
            let item = match item {
                Ok(p) => GalleryItem::from_proto(p, &s.crops_dir),
                Err(e) => {
                    tracing::warn!(gallery_id = c.gallery_id, error = %e, "gallery.Get failed, partial candidate");
                    partial_item(c, &s)
                }
            };
            Candidate {
                item,
                score: c.score,
                confidence: c.confidence,
                local_score: (c.local_score > 0.0).then_some(c.local_score),
                // Порог тот же, что применяет search (оба читают models/model.json).
                accepted: c.confidence >= s.threshold,
            }
        })
        .collect();

    Ok(Json(SearchResponse {
        id: resp.search_id,
        query_crop_url: crop_url(&resp.query_crop_uri, &s.crops_dir),
        threshold: s.threshold,
        candidates,
    }))
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct PatchMatch {
    /// Область на кропе запроса в координатах входа модели (inputWidth ×
    /// inputHeight из /api/info): кроп растягивается во вход целиком, так что
    /// в пиксели кропа — умножением на его ширину/высоту и делением на вход.
    pub query_region: Bbox,
    /// Область на кропе кандидата, в тех же координатах входа модели.
    pub candidate_region: Bbox,
    pub similarity: f32,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CompareResponse {
    pub matches: Vec<PatchMatch>,
    /// Доля взаимно совпавших патчей с весом по сходству, 0..1.
    pub local_score: f32,
    /// "few_shared_views", если ракурсы почти не пересекаются; иначе пусто.
    pub note: String,
    /// Теги кандидата из tagger; у query тегов нет — это не запись галереи.
    pub candidate_tags: Vec<Tag>,
}

/// Сверка пары: патч-токены query и кандидата из уже сохранённого поиска —
/// совпавшие области, local_score, теги кандидата. Требует модель с
/// патч-токенами (иначе search отвечает FailedPrecondition → 409).
#[utoipa::path(get, path = "/api/searches/{id}/compare/{gallery_id}", tag = "search",
    params(
        ("id" = i64, Path, description = "id поиска"),
        ("gallery_id" = i64, Path, description = "id кандидата в галерее")),
    responses(
        (status = 200, body = CompareResponse),
        (status = 404, body = crate::error::ErrorBody, description = "поиск или запись галереи не найдены"),
        (status = 409, body = crate::error::ErrorBody, description = "модель без патч-токенов"),
        (status = 502, body = crate::error::ErrorBody, description = "search недоступен")))]
pub async fn compare(
    State(s): State<AppState>,
    CompareIds(search_id, gallery_id): CompareIds,
) -> Result<Json<CompareResponse>, ApiError> {
    let resp = s
        .search
        .compare(search_id, gallery_id)
        .await
        .map_err(ApiError::search)?;
    Ok(Json(CompareResponse {
        matches: resp
            .matches
            .into_iter()
            .map(|m| PatchMatch {
                query_region: m.query_region.unwrap_or_default().into(),
                candidate_region: m.candidate_region.unwrap_or_default().into(),
                similarity: m.similarity,
            })
            .collect(),
        local_score: resp.local_score,
        note: resp.note,
        candidate_tags: resp
            .candidate_tags
            .into_iter()
            .map(|t| Tag {
                key: t.key,
                confidence: t.confidence,
                region: t.region.map(Into::into),
            })
            .collect(),
    }))
}

/// CSV поиска из search.Export: gallery_id,score,confidence,plate.
#[utoipa::path(get, path = "/api/searches/{id}/export.csv", tag = "search",
    params(("id" = i64, Path, description = "id поиска")),
    responses(
        (status = 200, content_type = "text/csv", description = "CSV, UTF-8"),
        (status = 404, body = crate::error::ErrorBody)))]
pub async fn export_csv(State(s): State<AppState>, Id(id): Id) -> Result<Response, ApiError> {
    let csv = s.search.export(id).await.map_err(ApiError::search)?;
    Ok((
        [
            (header::CONTENT_TYPE, "text/csv; charset=utf-8".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"search-{id}.csv\""),
            ),
        ],
        csv,
    )
        .into_response())
}
