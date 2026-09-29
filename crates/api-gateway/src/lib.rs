//! REST-шлюз для оператора и веб-клиента: валидация входа, OpenAPI,
//! маршрутизация по gRPC в gallery/search/inference. Своей БД нет. OpenAPI
//! генерируется из кода (utoipa), Swagger UI на /api/docs, спека коммитится в
//! docs/openapi.json (`just openapi`).

pub mod clients;
pub mod crops;
pub mod embed;
pub mod error;
pub mod extract;
pub mod gallery;
pub mod openapi;
pub mod searches;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Serialize;
use tonic::Status;
use tower_http::services::ServeDir;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use clients::{GalleryClient, InferenceClient, SearchClient};

/// Лимит тела запроса — в основном под CSV импорта; сам кадр ограничен
/// `gallery::MAX_FRAME_BYTES` (лимит сообщений у серверов gRPC).
const BODY_LIMIT: usize = 32 << 20;

#[derive(Clone)]
pub struct AppState {
    pub gallery: Arc<dyn GalleryClient>,
    pub search: Arc<dyn SearchClient>,
    pub inference: Arc<dyn InferenceClient>,
    /// Общий том кропов (только чтение): `file://` URI отдаются как /files/crops/....
    pub crops_dir: PathBuf,
    /// Корень датасета для импорта по CSV.
    pub dataset_dir: PathBuf,
    /// Порог отказа из models/model.json — для ответа /api/search; применяет его search.
    pub threshold: f32,
}

/// Маршруты шлюза. Swagger UI и /api/openapi.json — из той же спеки, что docs/openapi.json.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/info", get(info))
        .route("/api/embed", post(embed::embed))
        .route("/api/gallery", get(gallery::list).post(gallery::create))
        .route("/api/gallery/import", post(gallery::import))
        .route("/api/gallery/{id}", get(gallery::get_one))
        .route("/api/gallery/{id}/plate", put(gallery::set_plate))
        .route("/api/search", post(searches::search))
        .route(
            "/api/searches/{id}/compare/{gallery_id}",
            get(searches::compare),
        )
        .route("/api/searches/{id}/export.csv", get(searches::export_csv))
        .nest_service("/files/crops", ServeDir::new(&state.crops_dir))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
        .merge(SwaggerUi::new("/api/docs").url("/api/openapi.json", openapi::ApiDoc::openapi()))
}

#[derive(Serialize, ToSchema)]
pub struct Health {
    /// "ok" или "degraded" (хотя бы один из gRPC-сервисов не отвечает SERVING).
    pub status: &'static str,
    pub gallery: bool,
    pub search: bool,
    pub inference: bool,
}

fn serving(name: &str, r: Result<(), Status>) -> bool {
    if let Err(e) = &r {
        tracing::warn!(service = name, error = %e, "health check failed");
    }
    r.is_ok()
}

/// Готовность: 200, когда gallery, search и inference отвечают на gRPC health; иначе 503.
#[utoipa::path(get, path = "/api/health", tag = "service",
    responses((status = 200, body = Health), (status = 503, body = Health)))]
async fn health(State(s): State<AppState>) -> (StatusCode, Json<Health>) {
    let (g, se, i) = tokio::join!(s.gallery.health(), s.search.health(), s.inference.health());
    let (gallery, search, inference) = (
        serving("gallery", g),
        serving("search", se),
        serving("inference", i),
    );
    let ok = gallery && search && inference;
    let status = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = Health {
        status: if ok { "ok" } else { "degraded" },
        gallery,
        search,
        inference,
    };
    (status, Json(body))
}

#[derive(Serialize, ToSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub dim: i32,
    pub input_height: i32,
    pub input_width: i32,
    /// "cpu" | "cuda"
    pub execution_provider: String,
}

impl From<proto::InfoResponse> for ModelInfo {
    fn from(r: proto::InfoResponse) -> Self {
        ModelInfo {
            name: r.model_name,
            version: r.model_version,
            dim: r.dim,
            input_height: r.input_height,
            input_width: r.input_width,
            execution_provider: r.execution_provider,
        }
    }
}

#[derive(Serialize, ToSchema)]
pub struct Info {
    pub service: &'static str,
    /// Модель, загруженная сервисом inference; null, если он недоступен.
    pub model: Option<ModelInfo>,
    pub error: Option<String>,
}

/// Метаданные модели из сервиса inference (gRPC Info).
#[utoipa::path(get, path = "/api/info", tag = "service", responses((status = 200, body = Info)))]
async fn info(State(s): State<AppState>) -> Json<Info> {
    let (model, error) = match s.inference.info().await {
        Ok(r) => (Some(r.into()), None),
        Err(e) => (None, Some(error::ApiError::inference(e).message)),
    };
    Json(Info {
        service: "api-gateway",
        model,
        error,
    })
}
