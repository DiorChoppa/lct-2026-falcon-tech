//! `POST /api/embed` — прямой вызов inference.Embed без gallery и search:
//! именно эту операцию жюри может мерить по HTTP (04-architecture.md §4).

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use tonic::Status;
use utoipa::ToSchema;

use crate::error::ApiError;
use crate::extract::Form;
use crate::gallery::{to_proto_bbox, validate_frame};
use crate::AppState;

/// Поля формы POST /api/embed (multipart/form-data).
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct EmbedForm {
    /// Полный кадр JPEG/PNG, до 3 МиБ.
    #[schema(value_type = String, format = Binary)]
    image: Vec<u8>,
    /// JSON `{"x":..,"y":..,"w":..,"h":..}` в пикселях кадра.
    bbox: String,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct EmbedResponse {
    /// L2-нормированный вектор размерности dim.
    pub embedding: Vec<f32>,
    pub dim: usize,
    /// Чистое время инференса модели по данным inference, микросекунды.
    pub inference_us: i64,
}

/// Эмбеддинг одного bbox: кадр + bbox → inference.Embed.
#[utoipa::path(post, path = "/api/embed", tag = "service",
    request_body(content = EmbedForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, body = EmbedResponse),
        (status = 422, body = crate::error::ErrorBody, description = "bbox вне кадра, нет изображения"),
        (status = 502, body = crate::error::ErrorBody, description = "inference недоступен")))]
pub async fn embed(
    State(s): State<AppState>,
    Form(mut form): Form,
) -> Result<Json<EmbedResponse>, ApiError> {
    let (mut image, mut bbox) = (None, None);
    while let Some(field) = form.next_field().await? {
        match field.name().unwrap_or_default() {
            "image" => image = Some(field.bytes().await?.to_vec()),
            "bbox" => bbox = Some(field.text().await?),
            _ => {}
        }
    }
    let (image, bbox) = validate_frame(image, bbox)?;
    let resp = s
        .inference
        .embed(proto::EmbedRequest {
            frames: vec![proto::FrameWithBoxes {
                image,
                boxes: vec![to_proto_bbox(bbox)],
                frame_id: "0".into(),
            }],
            crops: vec![],
            with_patches: false,
        })
        .await
        .map_err(ApiError::inference)?;
    let embedding = resp
        .embeddings
        .into_iter()
        .next()
        .map(|e| e.values)
        .ok_or_else(|| ApiError::inference(Status::internal("пустой ответ Embed")))?;
    Ok(Json(EmbedResponse {
        dim: embedding.len(),
        embedding,
        inference_us: resp.inference_us,
    }))
}
