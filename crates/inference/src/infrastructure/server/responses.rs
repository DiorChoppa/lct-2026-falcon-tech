use std::collections::HashMap;
use std::error::Error;
use std::fmt::Display;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use getset::Getters;
use serde::Serialize;
use utoipa::ToSchema;

use super::constants::OK_MESSAGE;
use super::swagger::SwaggerExamples;

pub type ServerResult<T> = Result<T, ServerError>;

#[derive(Debug, Serialize, ToSchema)]
pub enum ServerError {
    InternalError(String),
    ServiceUnavailable,
}

impl Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::InternalError(msg) => write!(f, "Internal server error: {}", msg),
            ServerError::ServiceUnavailable => write!(f, "Service unavailable"),
        }
    }
}

impl Error for ServerError {}

impl ServerError {
    pub fn status_info(&self) -> (String, StatusCode) {
        match self {
            ServerError::InternalError(msg) => (msg.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
            ServerError::ServiceUnavailable => (
                "service unavailable".to_string(),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (message, status_code) = self.status_info();
        let response = HashMap::from([("message", message)]);
        let mut resp = Json(response).into_response();
        *resp.status_mut() = status_code;
        resp
    }
}

impl SwaggerExamples for ServerError {
    type Example = Self;

    fn example(value: Option<&str>) -> Self::Example {
        match value {
            None => ServerError::ServiceUnavailable,
            Some(msg) => ServerError::InternalError(msg.to_string()),
        }
    }
}

#[derive(Serialize, ToSchema, Getters)]
#[getset(get = "pub")]
pub struct Success {
    status: u16,
    message: String,
}

impl Default for Success {
    fn default() -> Self {
        Success {
            status: StatusCode::OK.as_u16(),
            message: OK_MESSAGE.to_owned(),
        }
    }
}

impl SwaggerExamples for Success {
    type Example = Self;

    fn example(value: Option<&str>) -> Self::Example {
        Success {
            status: StatusCode::OK.into(),
            message: value.unwrap_or(OK_MESSAGE).to_owned(),
        }
    }
}
