use axum::response::IntoResponse;
use axum::Json;

use crate::infrastructure::server::swagger::SwaggerExamples;

use super::constants::HEALTH_URL;
use super::responses::{ServerError, ServerResult, Success};

#[utoipa::path(
    get,
    path = HEALTH_URL,
    tag = "health",
    responses(
        (
            status = 200,
            body = Success,
            content_type="application/json",
            description = "Health response of service",
        ),
        (
            status = 503,
            body = ServerError,
            description = "Service is not available",
            example = json!(ServerError::example(None)),
        ),
    ),
)]
pub async fn health() -> ServerResult<impl IntoResponse> {
    Ok(Json(Success::default()))
}
