use std::sync::Arc;

use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};
use proto::search_server::Search;
use proto::{CompareRequest, ExportRequest, SearchRequest};
use tonic::{Code, Request};

use crate::application::repositories::CandidateRow;
use crate::application::use_cases::UseCases;
use crate::domain::ThresholdManifest;
use crate::infrastructure::grpc::service::SearchGrpcService;
use crate::test_utils::{MockEmbeddingClient, MockGalleryReadRepository, MockSearchRepository};

fn solid_png() -> Vec<u8> {
    let img = RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&img, 4, 4, ExtendedColorType::Rgb8)
        .unwrap();
    out
}

fn service(gallery_read: MockGalleryReadRepository) -> SearchGrpcService {
    let use_cases = UseCases::new(
        Arc::new(gallery_read),
        Arc::new(MockSearchRepository::default()),
        Arc::new(storage::test_utils::MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
        ThresholdManifest {
            threshold: 0.5,
            alpha: 1.0,
        },
    );
    SearchGrpcService::new(Arc::new(use_cases))
}

#[tokio::test]
async fn search_returns_candidates_above_threshold() {
    let service = service(MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.9,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: Some("A123BC".into()),
    }]));

    let response = service
        .search(Request::new(SearchRequest {
            image: solid_png(),
            bbox: Some(proto::BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            top_n: 10,
            details: false,
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(response.accepted);
    assert_eq!(1, response.candidates.len());
    assert_eq!("A123BC", response.candidates[0].plate);
}

#[tokio::test]
async fn search_rejects_missing_bbox() {
    let service = service(MockGalleryReadRepository::default());

    let status = service
        .search(Request::new(SearchRequest {
            image: solid_png(),
            bbox: None,
            top_n: 10,
            details: false,
        }))
        .await
        .unwrap_err();

    assert_eq!(Code::InvalidArgument, status.code());
}

#[tokio::test]
async fn compare_returns_not_found_for_unknown_search() {
    let service = service(MockGalleryReadRepository::default());

    let status = service
        .compare(Request::new(CompareRequest {
            search_id: 404,
            gallery_id: 1,
        }))
        .await
        .unwrap_err();

    assert_eq!(Code::NotFound, status.code());
}

#[tokio::test]
async fn export_returns_not_found_for_unknown_search() {
    let service = service(MockGalleryReadRepository::default());

    let status = service
        .export(Request::new(ExportRequest { search_id: 404 }))
        .await
        .unwrap_err();

    assert_eq!(Code::NotFound, status.code());
}

#[tokio::test]
async fn export_returns_csv_for_a_search_just_made() {
    let service = service(MockGalleryReadRepository::with_rows(vec![CandidateRow {
        gallery_id: 1,
        score: 0.9,
        crop_uri: "mock://bucket/1.jpg".into(),
        plate: None,
    }]));
    let search = service
        .search(Request::new(SearchRequest {
            image: solid_png(),
            bbox: Some(proto::BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            top_n: 10,
            details: false,
        }))
        .await
        .unwrap()
        .into_inner();

    let response = service
        .export(Request::new(ExportRequest {
            search_id: search.search_id,
        }))
        .await
        .unwrap()
        .into_inner();

    let csv = String::from_utf8(response.csv).unwrap();
    assert!(csv.starts_with("gallery_id,score,confidence,plate\n"));
}
