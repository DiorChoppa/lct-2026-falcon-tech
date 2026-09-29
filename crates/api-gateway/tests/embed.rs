//! POST /api/embed — прямой вызов inference для замера.

mod common;

use axum::http::StatusCode;
use common::{ctx, ctx_with, jpeg, multipart, send, FakeSearch, DIM};

fn embed_req(bbox: &str) -> axum::http::Request<axum::body::Body> {
    multipart(
        "/api/embed",
        &[("bbox", bbox)],
        ("image", "f.jpg", &jpeg(100, 100)),
    )
}

#[tokio::test]
async fn embed_returns_vector_and_timing() {
    let c = ctx();
    let (status, r, _) = send(&c.state, embed_req(r#"{"x":0,"y":0,"w":50,"h":50}"#)).await;
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_eq!(r["dim"], DIM);
    assert_eq!(r["embedding"].as_array().unwrap().len(), DIM);
    assert_eq!(r["embedding"][0], 1.0);
    assert_eq!(r["inferenceUs"], 1234);
}

#[tokio::test]
async fn embed_validates_bbox_before_calling_inference() {
    let c = ctx();
    let (status, r, _) = send(&c.state, embed_req(r#"{"x":90,"y":0,"w":50,"h":50}"#)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r["error"]["code"], "VALIDATION_ERROR");
}

#[tokio::test]
async fn embed_reports_inference_outage_as_502() {
    let c = ctx_with(false, FakeSearch::default(), true);
    let (status, r, _) = send(&c.state, embed_req(r#"{"x":0,"y":0,"w":50,"h":50}"#)).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(r["error"]["code"], "INFERENCE_UNAVAILABLE");
}
