mod constants;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use tower::ServiceExt;

use crate::infrastructure::server::{
    init_router,
    responses::{ServerError, Success},
    swagger::SwaggerExamples,
};
use constants::UNUSUAL_STR;

use super::constants::{HEALTH_URL, OK_MESSAGE};

#[test]
fn test_server_error_internal_error() {
    let expected = "test message";
    let error = ServerError::InternalError(expected.to_string());
    let (message, status_code) = error.status_info();
    let response = error.into_response();

    assert_eq!(expected, message);
    assert_eq!(StatusCode::INTERNAL_SERVER_ERROR, status_code);
    assert_eq!(StatusCode::INTERNAL_SERVER_ERROR, response.status());
}

#[test]
fn test_server_error_service_unavailable() {
    let error = ServerError::ServiceUnavailable;
    let (message, status_code) = error.status_info();
    let response = error.into_response();

    assert_eq!("service unavailable", message);
    assert_eq!(StatusCode::SERVICE_UNAVAILABLE, status_code);
    assert_eq!(StatusCode::SERVICE_UNAVAILABLE, response.status());
}

#[test]
fn test_success_default_success() {
    let success = Success::default();

    assert_eq!(&StatusCode::OK.as_u16(), success.status());
    assert_eq!(&OK_MESSAGE, success.message());
}

#[test]
fn test_swagger_success_example() {
    let example = Success::example(Some(UNUSUAL_STR));

    assert_eq!(&StatusCode::OK.as_u16(), example.status());
    assert_eq!(&UNUSUAL_STR, example.message());
}

#[test]
fn test_swagger_server_error_example() {
    let example = ServerError::example(Some(UNUSUAL_STR));

    assert_eq!(
        (UNUSUAL_STR.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
        example.status_info()
    );
}

#[tokio::test]
async fn test_health_route_returns_ok() {
    let router = init_router();
    let request = Request::builder()
        .uri(HEALTH_URL)
        .body(Body::empty())
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    assert_eq!(StatusCode::OK, response.status());
}
