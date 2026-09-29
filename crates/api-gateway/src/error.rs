//! Единый формат ошибок: `{ "error": { "code": "...", "message": "..." } }`.
//! Ошибки gRPC-сервисов приводятся к HTTP в одном месте (`upstream`).

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tonic::{Code, Status};
use utoipa::ToSchema;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

/// Кто из сервисов вернул ошибку — от этого зависит код `*_UNAVAILABLE`.
#[derive(Debug, Clone, Copy)]
pub enum Upstream {
    Gallery,
    Search,
    Inference,
}

impl Upstream {
    fn name(self) -> &'static str {
        match self {
            Upstream::Gallery => "gallery",
            Upstream::Search => "search",
            Upstream::Inference => "inference",
        }
    }

    fn unavailable_code(self) -> &'static str {
        match self {
            Upstream::Gallery => "GALLERY_UNAVAILABLE",
            Upstream::Search => "SEARCH_UNAVAILABLE",
            Upstream::Inference => "INFERENCE_UNAVAILABLE",
        }
    }
}

impl ApiError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "VALIDATION_ERROR",
            message: message.into(),
        }
    }

    /// tonic::Code → HTTP. Сообщение сервиса пробрасывается как есть: web
    /// показывает его оператору. Cancelled — так tonic сообщает о клиентском
    /// таймауте (`Request::set_timeout`), поэтому он равен недоступности.
    pub fn upstream(service: Upstream, status: Status) -> Self {
        let message = status.message().to_string();
        match status.code() {
            Code::InvalidArgument => Self::validation(message),
            Code::NotFound => Self {
                status: StatusCode::NOT_FOUND,
                code: "NOT_FOUND",
                message,
            },
            Code::Unavailable | Code::DeadlineExceeded | Code::Cancelled => {
                tracing::warn!(service = service.name(), error = %message, "upstream unavailable");
                Self {
                    status: StatusCode::BAD_GATEWAY,
                    code: service.unavailable_code(),
                    message: format!("сервис {} недоступен: {message}", service.name()),
                }
            }
            Code::FailedPrecondition => Self {
                status: StatusCode::CONFLICT,
                code: "UPSTREAM_PRECONDITION",
                message,
            },
            _ => {
                tracing::warn!(service = service.name(), error = %message, "upstream error");
                Self {
                    status: StatusCode::BAD_GATEWAY,
                    code: "UPSTREAM_ERROR",
                    message: format!("сервис {}: {message}", service.name()),
                }
            }
        }
    }

    pub fn gallery(status: Status) -> Self {
        Self::upstream(Upstream::Gallery, status)
    }

    pub fn search(status: Status) -> Self {
        Self::upstream(Upstream::Search, status)
    }

    pub fn inference(status: Status) -> Self {
        Self::upstream(Upstream::Inference, status)
    }
}

impl From<axum::extract::multipart::MultipartError> for ApiError {
    fn from(e: axum::extract::multipart::MultipartError) -> Self {
        Self::validation(format!("multipart: {e}"))
    }
}

/// Тело ошибки для OpenAPI.
#[derive(Serialize, ToSchema)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Serialize, ToSchema)]
pub struct ErrorDetail {
    /// Машиночитаемый код: VALIDATION_ERROR, NOT_FOUND, GALLERY_UNAVAILABLE,
    /// SEARCH_UNAVAILABLE, INFERENCE_UNAVAILABLE, UPSTREAM_PRECONDITION,
    /// UPSTREAM_ERROR
    pub code: String,
    pub message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                code: self.code.into(),
                message: self.message,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(service: Upstream, status: Status) -> (StatusCode, &'static str) {
        let e = ApiError::upstream(service, status);
        (e.status, e.code)
    }

    #[test]
    fn grpc_codes_map_to_http() {
        assert_eq!(
            map(Upstream::Gallery, Status::invalid_argument("bbox")),
            (StatusCode::UNPROCESSABLE_ENTITY, "VALIDATION_ERROR")
        );
        assert_eq!(
            map(Upstream::Gallery, Status::not_found("nope")),
            (StatusCode::NOT_FOUND, "NOT_FOUND")
        );
        assert_eq!(
            map(Upstream::Gallery, Status::unavailable("down")),
            (StatusCode::BAD_GATEWAY, "GALLERY_UNAVAILABLE")
        );
        assert_eq!(
            map(Upstream::Search, Status::cancelled("Timeout expired")),
            (StatusCode::BAD_GATEWAY, "SEARCH_UNAVAILABLE")
        );
        assert_eq!(
            map(Upstream::Inference, Status::deadline_exceeded("slow")),
            (StatusCode::BAD_GATEWAY, "INFERENCE_UNAVAILABLE")
        );
        assert_eq!(
            map(Upstream::Search, Status::failed_precondition("no patches")),
            (StatusCode::CONFLICT, "UPSTREAM_PRECONDITION")
        );
        assert_eq!(
            map(Upstream::Gallery, Status::internal("boom")),
            (StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR")
        );
    }

    #[test]
    fn upstream_message_is_passed_through() {
        let e = ApiError::gallery(Status::not_found("gallery item not found"));
        assert_eq!(e.message, "gallery item not found");
        let e = ApiError::inference(Status::unavailable("connection refused"));
        assert!(e.message.contains("connection refused"), "{}", e.message);
    }
}
