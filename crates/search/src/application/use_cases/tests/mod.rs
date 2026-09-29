mod fixtures;

use std::sync::Arc;

use common::BBox;
use inference_client::{Embedded, ModelInfo};

use crate::application::repositories::CandidateRow;
use crate::application::use_cases::UseCases;
use crate::domain::ThresholdManifest;
use crate::test_utils::{MockEmbeddingClient, MockGalleryReadRepository, MockSearchRepository};
use fixtures::solid_png;

fn bbox() -> BBox {
    BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    }
}

fn use_cases(
    gallery_read: MockGalleryReadRepository,
    embedding: MockEmbeddingClient,
    threshold: ThresholdManifest,
) -> UseCases {
    UseCases::new(
        Arc::new(gallery_read),
        Arc::new(MockSearchRepository::default()),
        Arc::new(storage::test_utils::MockObjectRepository::default()),
        Arc::new(embedding),
        threshold,
    )
}

fn threshold(value: f32) -> ThresholdManifest {
    ThresholdManifest {
        threshold: value,
        alpha: 1.0,
    }
}

#[tokio::test]
async fn search_accepts_candidates_at_or_above_threshold() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![
        CandidateRow {
            gallery_id: 1,
            score: 0.9,
            crop_uri: "mock://bucket/1.jpg".into(),
            plate: Some("A123BC".into()),
        },
        CandidateRow {
            gallery_id: 2,
            score: 0.2,
            crop_uri: "mock://bucket/2.jpg".into(),
            plate: None,
        },
    ]);
    let use_cases = use_cases(gallery_read, MockEmbeddingClient::default(), threshold(0.5));

    let record = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    assert!(record.accepted);
    // Ниже порога тоже возвращается: accepted — флаг поиска, не фильтр.
    assert_eq!(2, record.candidates.len());
    assert_eq!(1, record.candidates[0].gallery_id);
    assert_eq!(2, record.candidates[1].gallery_id);
    assert_eq!(None, record.candidates[0].local_score);
}

#[tokio::test]
async fn search_rejects_when_no_candidate_passes_threshold() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.1,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: None,
    }]);
    let use_cases = use_cases(gallery_read, MockEmbeddingClient::default(), threshold(0.5));

    let record = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    assert!(!record.accepted);
    assert_eq!(1, record.candidates.len());
    assert_eq!(1, record.candidates[0].gallery_id);
}

#[tokio::test]
async fn search_orders_candidates_by_confidence() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![
        CandidateRow {
            gallery_id: 1,
            score: 0.3,
            crop_uri: "mock://bucket/1.jpg".into(),
            plate: None,
        },
        CandidateRow {
            gallery_id: 2,
            score: 0.8,
            crop_uri: "mock://bucket/2.jpg".into(),
            plate: None,
        },
    ]);
    let use_cases = use_cases(gallery_read, MockEmbeddingClient::default(), threshold(0.5));

    let record = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    let ids: Vec<i64> = record.candidates.iter().map(|c| c.gallery_id).collect();
    assert_eq!(vec![2, 1], ids);
}

#[tokio::test]
async fn search_writes_the_query_crop_before_returning() {
    let gallery_read = MockGalleryReadRepository::default();
    let object_repository = Arc::new(storage::test_utils::MockObjectRepository::default());
    let use_cases = UseCases::new(
        Arc::new(gallery_read),
        Arc::new(MockSearchRepository::default()),
        object_repository.clone(),
        Arc::new(MockEmbeddingClient::default()),
        threshold(0.5),
    );

    let record = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    assert_eq!(1, object_repository.received_puts().len());
    assert!(!record.query_crop_uri.is_empty());
}

#[tokio::test]
async fn search_computes_local_score_when_details_requested_and_model_supports_patches() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.6,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: None,
    }]);
    let embedding = MockEmbeddingClient::with_info(ModelInfo {
        model_version: "v1".into(),
        dim: 3,
        input_height: 4,
        input_width: 4,
        supports_patches: true,
        patch_grid_h: 1,
        patch_grid_w: 1,
        patch_dim: 2,
    })
    .with_embedded_by_uri(vec![
        Embedded {
            embedding: vec![],
            patches: Some(vec![1.0, 0.0]),
        },
        Embedded {
            embedding: vec![],
            patches: Some(vec![1.0, 0.0]),
        },
    ]);
    let use_cases = use_cases(gallery_read, embedding, threshold(0.0));

    let record = use_cases
        .search(solid_png(), bbox(), 10, true)
        .await
        .unwrap();

    assert_eq!(Some(1.0), record.candidates[0].local_score);
}

#[tokio::test]
async fn search_skips_local_score_when_model_lacks_patches() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.6,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: None,
    }]);
    let use_cases = use_cases(gallery_read, MockEmbeddingClient::default(), threshold(0.0));

    let record = use_cases
        .search(solid_png(), bbox(), 10, true)
        .await
        .unwrap();

    assert_eq!(None, record.candidates[0].local_score);
    assert_eq!(0.6, record.candidates[0].confidence);
}

#[tokio::test]
async fn compare_errors_when_model_lacks_patches() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.6,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: None,
    }]);
    let use_cases = use_cases(gallery_read, MockEmbeddingClient::default(), threshold(0.0));
    let search = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    let err = use_cases.compare(search.id, 1).await.unwrap_err();

    assert!(err.to_string().contains("no patches output"));
}

#[tokio::test]
async fn compare_returns_matches_local_score_and_candidate_tags() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.6,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: None,
    }]);
    let embedding = MockEmbeddingClient::with_info(ModelInfo {
        model_version: "v1".into(),
        dim: 3,
        input_height: 4,
        input_width: 4,
        supports_patches: true,
        patch_grid_h: 1,
        patch_grid_w: 1,
        patch_dim: 2,
    })
    .with_embedded_by_uri(vec![
        Embedded {
            embedding: vec![],
            patches: Some(vec![1.0, 0.0]),
        },
        Embedded {
            embedding: vec![],
            patches: Some(vec![1.0, 0.0]),
        },
    ]);
    let mut gallery_read = gallery_read;
    gallery_read.tags.insert(
        1,
        vec![crate::domain::Tag {
            key: "roof_box".into(),
            confidence: 0.9,
            region: None,
        }],
    );
    let use_cases = use_cases(gallery_read, embedding, threshold(0.0));
    let search = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    let result = use_cases.compare(search.id, 1).await.unwrap();

    assert_eq!(1, result.matches.len());
    assert!((result.local_score - 1.0).abs() < 1e-6);
    assert_eq!(1, result.candidate_tags.len());
    assert_eq!("roof_box", result.candidate_tags[0].key);
}

#[tokio::test]
async fn compare_errors_for_unknown_search_id() {
    let use_cases = use_cases(
        MockGalleryReadRepository::default(),
        MockEmbeddingClient::default(),
        threshold(0.5),
    );

    let err = use_cases.compare(404, 1).await.unwrap_err();

    assert!(err.to_string().contains("not found"));
}

#[tokio::test]
async fn export_csv_writes_header_and_candidate_rows() {
    let gallery_read = MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.9,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: Some("A123BC".into()),
    }]);
    let use_cases = use_cases(gallery_read, MockEmbeddingClient::default(), threshold(0.5));
    let search = use_cases
        .search(solid_png(), bbox(), 10, false)
        .await
        .unwrap();

    let csv = use_cases.export_csv(search.id).await.unwrap();
    let csv = String::from_utf8(csv).unwrap();

    assert!(csv.starts_with("gallery_id,score,confidence,plate\n"));
    assert!(csv.contains("1,0.9,0.9,A123BC"));
}

#[tokio::test]
async fn export_csv_errors_for_unknown_search_id() {
    let use_cases = use_cases(
        MockGalleryReadRepository::default(),
        MockEmbeddingClient::default(),
        threshold(0.5),
    );

    let err = use_cases.export_csv(404).await.unwrap_err();

    assert!(err.to_string().contains("not found"));
}
