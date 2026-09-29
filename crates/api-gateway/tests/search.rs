//! Поиск через шлюз: кандидаты search + карточки из gallery, отказ,
//! устойчивость к недоступному gallery, экспорт CSV как есть.

mod common;

use std::sync::Arc;

use axum::http::{header, StatusCode};
use common::{ctx, ctx_with, get, jpeg, multipart, post_gallery, send, send_raw, FakeSearch};

fn search_req(bbox: &str, extra: &[(&str, &str)]) -> axum::http::Request<axum::body::Body> {
    let mut fields = vec![("bbox", bbox)];
    fields.extend_from_slice(extra);
    multipart("/api/search", &fields, ("image", "f.jpg", &jpeg(400, 100)))
}

fn candidate(
    gallery_id: i64,
    score: f32,
    plate: &str,
    crops_root: &std::path::Path,
) -> proto::Candidate {
    proto::Candidate {
        gallery_id,
        score,
        confidence: score - 0.05,
        local_score: 0.0,
        plate: plate.into(),
        crop_uri: format!(
            "file://{}/gallery-crops/c{gallery_id}.jpg",
            crops_root.display()
        ),
    }
}

/// Два кандидата, прошедших порог, кроп запроса — на общем томе.
fn found(crops_root: &std::path::Path) -> FakeSearch {
    FakeSearch {
        response: proto::SearchResponse {
            search_id: 42,
            query_crop_uri: format!("file://{}/query-crops/q.jpg", crops_root.display()),
            candidates: vec![
                candidate(1, 0.95, "A001AA77", crops_root),
                candidate(2, 0.61, "", crops_root),
            ],
            accepted: true,
        },
        export_id: 42,
        ..FakeSearch::default()
    }
}

#[tokio::test]
async fn search_returns_candidates_with_gallery_items() {
    let mut c = ctx();
    c.state.search = Arc::new(found(&c.state.crops_dir));
    // Записи 1 и 2 в фейковой галерее — из них берутся карточки.
    post_gallery(&c.state, r#"{"x":0,"y":0,"w":40,"h":30}"#).await;
    post_gallery(&c.state, r#"{"x":10,"y":0,"w":40,"h":30}"#).await;

    let (status, r, _) = send(&c.state, search_req(r#"{"x":10,"y":0,"w":40,"h":30}"#, &[])).await;
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_eq!(r["id"], 42);
    assert_eq!(r["threshold"], 0.5);
    assert_eq!(r["queryCropUrl"], "/files/crops/query-crops/q.jpg");
    let cands = r["candidates"].as_array().unwrap();
    assert_eq!(cands.len(), 2);
    assert_eq!(cands[0]["item"]["id"], 1);
    assert_eq!(cands[0]["item"]["imageId"], "gen-1");
    assert_eq!(cands[0]["item"]["vehicleId"], "403");
    assert_eq!(
        cands[0]["item"]["cropUrl"],
        "/files/crops/gallery-crops/gen-1.jpg"
    );
    assert!((cands[0]["score"].as_f64().unwrap() - 0.95).abs() < 1e-6);
    assert!((cands[0]["confidence"].as_f64().unwrap() - 0.90).abs() < 1e-6);
    assert_eq!(cands[0]["accepted"], true);
    assert!(cands[0]["localScore"].is_null());
    assert_eq!(cands[1]["item"]["id"], 2);
    assert_eq!(cands[1]["accepted"], true);
}

#[tokio::test]
async fn search_marks_each_candidate_against_threshold() {
    let mut c = ctx();
    let crops = c.state.crops_dir.clone();
    c.state.search = Arc::new(FakeSearch {
        response: proto::SearchResponse {
            search_id: 7,
            query_crop_uri: format!("file://{}/query-crops/q.jpg", crops.display()),
            // confidence = score - 0.05: 0.85 проходит порог 0.5, 0.40 — нет.
            candidates: vec![
                candidate(1, 0.90, "", &crops),
                candidate(2, 0.45, "", &crops),
            ],
            accepted: true,
        },
        ..FakeSearch::default()
    });
    let (status, r, _) = send(&c.state, search_req(r#"{"x":10,"y":0,"w":40,"h":30}"#, &[])).await;
    assert_eq!(status, StatusCode::OK, "{r}");
    let cands = r["candidates"].as_array().unwrap();
    assert_eq!(cands.len(), 2);
    assert_eq!(cands[0]["accepted"], true);
    assert_eq!(cands[1]["accepted"], false);
}

#[tokio::test]
async fn search_refusal_has_no_accepted_candidates() {
    let c = ctx();
    let (status, r, _) = send(&c.state, search_req(r#"{"x":10,"y":0,"w":40,"h":30}"#, &[])).await;
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_eq!(r["candidates"].as_array().unwrap().len(), 0);
    assert_eq!(r["threshold"], 0.5);
}

#[tokio::test]
async fn search_survives_gallery_outage_with_partial_items() {
    let mut c = ctx_with(true, FakeSearch::default(), false);
    c.state.search = Arc::new(found(&c.state.crops_dir));
    let (status, r, _) = send(&c.state, search_req(r#"{"x":10,"y":0,"w":40,"h":30}"#, &[])).await;
    assert_eq!(status, StatusCode::OK, "{r}");
    let cands = r["candidates"].as_array().unwrap();
    assert_eq!(cands.len(), 2);
    assert_eq!(cands[0]["item"]["id"], 1);
    assert_eq!(cands[0]["item"]["plate"], "A001AA77");
    assert_eq!(
        cands[0]["item"]["cropUrl"],
        "/files/crops/gallery-crops/c1.jpg"
    );
    assert_eq!(cands[0]["item"]["imageId"], "");
    assert!(cands[1]["item"]["plate"].is_null());
}

#[tokio::test]
async fn search_validates_input() {
    let c = ctx();
    let (status, r, _) = send(
        &c.state,
        search_req(r#"{"x":390,"y":0,"w":40,"h":30}"#, &[]),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r["error"]["code"], "VALIDATION_ERROR");

    let (status, r, _) = send(
        &c.state,
        search_req(r#"{"x":0,"y":0,"w":40,"h":30}"#, &[("top_n", "ten")]),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r["error"]["code"], "VALIDATION_ERROR");
}

#[tokio::test]
async fn search_reports_search_outage_as_502() {
    let c = ctx_with(
        false,
        FakeSearch {
            down: true,
            ..FakeSearch::default()
        },
        false,
    );
    let (status, r, _) = send(&c.state, search_req(r#"{"x":10,"y":0,"w":40,"h":30}"#, &[])).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(r["error"]["code"], "SEARCH_UNAVAILABLE");
}

#[tokio::test]
async fn compare_returns_matches_and_tags() {
    let mut c = ctx();
    c.state.search = Arc::new(FakeSearch {
        compare_response: proto::CompareResponse {
            matches: vec![proto::PatchMatch {
                query_region: Some(proto::Region {
                    x: 1,
                    y: 2,
                    w: 10,
                    h: 12,
                }),
                candidate_region: Some(proto::Region {
                    x: 3,
                    y: 4,
                    w: 8,
                    h: 9,
                }),
                similarity: 0.87,
            }],
            local_score: 0.42,
            note: String::new(),
            candidate_tags: vec![proto::Tag {
                key: "roof_box".into(),
                confidence: 0.9,
                region: None,
            }],
        },
        ..FakeSearch::default()
    });

    let (status, r, _) = send(&c.state, get("/api/searches/7/compare/1")).await;

    assert_eq!(status, StatusCode::OK, "{r}");
    assert!((r["localScore"].as_f64().unwrap() - 0.42).abs() < 1e-6);
    let matches = r["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["queryRegion"]["x"], 1);
    assert_eq!(matches[0]["candidateRegion"]["w"], 8);
    assert!((matches[0]["similarity"].as_f64().unwrap() - 0.87).abs() < 1e-6);
    assert_eq!(r["candidateTags"][0]["key"], "roof_box");
}

#[tokio::test]
async fn compare_returns_not_found_for_unknown_ids() {
    let c = ctx();
    let (status, r, _) = send(&c.state, get("/api/searches/999/compare/1")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(r["error"]["code"], "NOT_FOUND");
}

#[tokio::test]
async fn compare_reports_missing_patch_support_as_conflict() {
    struct FailingCompare;
    #[async_trait::async_trait]
    impl api_gateway::clients::SearchClient for FailingCompare {
        async fn search(
            &self,
            _req: proto::SearchRequest,
        ) -> Result<proto::SearchResponse, tonic::Status> {
            unimplemented!()
        }
        async fn compare(
            &self,
            _search_id: i64,
            _gallery_id: i64,
        ) -> Result<proto::CompareResponse, tonic::Status> {
            Err(tonic::Status::failed_precondition(
                "model has no patches output",
            ))
        }
        async fn export(&self, _search_id: i64) -> Result<Vec<u8>, tonic::Status> {
            unimplemented!()
        }
        async fn health(&self) -> Result<(), tonic::Status> {
            Ok(())
        }
    }
    let mut c = ctx();
    c.state.search = Arc::new(FailingCompare);

    let (status, r, _) = send(&c.state, get("/api/searches/7/compare/1")).await;

    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(r["error"]["code"], "UPSTREAM_PRECONDITION");
}

#[tokio::test]
async fn export_passes_search_csv_through() {
    let c = ctx();
    let (status, headers, bytes) = send_raw(&c.state, get("/api/searches/7/export.csv")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("text/csv"));
    assert_eq!(
        headers[header::CONTENT_DISPOSITION],
        "attachment; filename=\"search-7.csv\""
    );
    assert_eq!(bytes, FakeSearch::default().csv);

    let (status, body, _) = send(&c.state, get("/api/searches/999999/export.csv")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "NOT_FOUND");
}
