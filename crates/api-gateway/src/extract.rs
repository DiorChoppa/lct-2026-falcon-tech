//! Экстракторы axum с ошибками в контрактном формате `{error:{code,message}}`:
//! стандартные `Path`/`Json`/`Multipart` при некорректном входе отвечают
//! text/plain 400, а OpenAPI обещает единый JSON.

use axum::extract::{FromRequest, FromRequestParts, Multipart, Path, Request};
use axum::http::request::Parts;
use axum::Json;
use serde::de::DeserializeOwned;

use crate::error::ApiError;

/// `{id}` из пути.
pub struct Id(pub i64);

impl<S: Send + Sync> FromRequestParts<S> for Id {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, ApiError> {
        let Path(id) = Path::<i64>::from_request_parts(parts, state)
            .await
            .map_err(|e| ApiError::validation(format!("id: {}", e.body_text())))?;
        Ok(Id(id))
    }
}

/// `{id}/compare/{gallery_id}` из пути.
pub struct CompareIds(pub i64, pub i64);

impl<S: Send + Sync> FromRequestParts<S> for CompareIds {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, ApiError> {
        let Path((id, gallery_id)) = Path::<(i64, i64)>::from_request_parts(parts, state)
            .await
            .map_err(|e| ApiError::validation(format!("id: {}", e.body_text())))?;
        Ok(CompareIds(id, gallery_id))
    }
}

/// JSON-тело.
pub struct JsonBody<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequest<S> for JsonBody<T> {
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, ApiError> {
        let Json(body) = Json::<T>::from_request(req, state)
            .await
            .map_err(|e| ApiError::validation(format!("json: {}", e.body_text())))?;
        Ok(JsonBody(body))
    }
}

/// multipart/form-data.
pub struct Form(pub Multipart);

impl<S: Send + Sync> FromRequest<S> for Form {
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, ApiError> {
        let form = Multipart::from_request(req, state)
            .await
            .map_err(|e| ApiError::validation(format!("multipart: {}", e.body_text())))?;
        Ok(Form(form))
    }
}
