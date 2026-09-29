//! Состояние шлюза: health по gRPC-зависимостям, info из inference, Swagger.

mod common;

use axum::http::StatusCode;
use common::{ctx, ctx_with, get, send, FakeSearch};

#[tokio::test]
async fn health_is_ok_when_all_services_serve() {
    let c = ctx();
    let (status, body, _) = send(&c.state, get("/api/health")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "ok");
    assert_eq!(body["gallery"], true);
    assert_eq!(body["search"], true);
    assert_eq!(body["inference"], true);
}

#[tokio::test]
async fn health_is_503_when_a_service_is_down() {
    let c = ctx_with(true, FakeSearch::default(), false);
    let (status, body, _) = send(&c.state, get("/api/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["gallery"], false);
    assert_eq!(body["search"], true);

    let c = ctx_with(false, FakeSearch::default(), true);
    let (status, body, _) = send(&c.state, get("/api/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["inference"], false);
}

#[tokio::test]
async fn info_returns_model_from_inference() {
    let c = ctx();
    let (status, body, _) = send(&c.state, get("/api/info")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["service"], "api-gateway");
    assert_eq!(body["model"]["name"], "placeholder");
    assert_eq!(body["model"]["version"], "0.0.0");
    assert_eq!(body["model"]["dim"], common::DIM);
    assert_eq!(body["model"]["executionProvider"], "cpu");
    assert!(body["error"].is_null());
}

#[tokio::test]
async fn info_is_200_with_error_when_inference_is_down() {
    let c = ctx_with(false, FakeSearch::default(), true);
    let (status, body, _) = send(&c.state, get("/api/info")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["model"].is_null());
    assert!(body["error"].as_str().unwrap().contains("inference"));
}

#[tokio::test]
async fn openapi_json_is_served() {
    let c = ctx();
    let (status, body, _) = send(&c.state, get("/api/openapi.json")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["paths"]["/api/health"].is_object());
    assert!(body["paths"]["/api/embed"].is_object());
}

#[tokio::test]
async fn swagger_ui_is_served() {
    let c = ctx();
    let (status, _, _) = send(&c.state, get("/api/docs/")).await;
    assert_eq!(status, StatusCode::OK);
}
