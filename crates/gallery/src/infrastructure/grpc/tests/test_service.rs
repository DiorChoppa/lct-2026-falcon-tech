use std::sync::Arc;

use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};
use proto::gallery_server::Gallery;
use proto::{AddRequest, DeleteRequest, GetRequest, ListRequest, SetPlateRequest, SetTagsRequest};
use storage::test_utils::MockObjectRepository;
use tonic::{Code, Request};

use crate::application::use_cases::UseCases;
use crate::infrastructure::grpc::service::GalleryGrpcService;
use crate::test_utils::{MockEmbeddingClient, MockGalleryRepository, MockTaggerClient};

fn solid_png() -> Vec<u8> {
    let img = RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&img, 4, 4, ExtendedColorType::Rgb8)
        .unwrap();
    out
}

fn service() -> GalleryGrpcService {
    let use_cases = UseCases::new(
        Arc::new(MockGalleryRepository::default()),
        Arc::new(MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
        Arc::new(MockTaggerClient::default()),
    );
    GalleryGrpcService::new(Arc::new(use_cases))
}

#[tokio::test]
async fn add_returns_a_new_item_with_bbox_and_model_version() {
    let service = service();

    let response = service
        .add(Request::new(AddRequest {
            image: solid_png(),
            bbox: Some(proto::BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            image_id: String::new(),
            vehicle_id: "v1".into(),
            plate: String::new(),
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(!response.image_id.is_empty());
    assert_eq!("v1", response.vehicle_id);
    assert_eq!("v1", response.model_version);
    assert_eq!(
        Some(proto::BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4
        }),
        response.bbox
    );
}

#[tokio::test]
async fn add_rejects_missing_bbox() {
    let service = service();

    let status = service
        .add(Request::new(AddRequest {
            image: solid_png(),
            bbox: None,
            image_id: String::new(),
            vehicle_id: String::new(),
            plate: String::new(),
        }))
        .await
        .unwrap_err();

    assert_eq!(Code::InvalidArgument, status.code());
}

#[tokio::test]
async fn get_returns_not_found_for_missing_id() {
    let service = service();

    let status = service
        .get(Request::new(GetRequest { id: 404 }))
        .await
        .unwrap_err();

    assert_eq!(Code::NotFound, status.code());
}

#[tokio::test]
async fn get_returns_the_item_added_earlier() {
    let service = service();
    let added = service
        .add(Request::new(AddRequest {
            image: solid_png(),
            bbox: Some(proto::BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            image_id: "frame-1".into(),
            vehicle_id: String::new(),
            plate: String::new(),
        }))
        .await
        .unwrap()
        .into_inner();

    let got = service
        .get(Request::new(GetRequest { id: added.id }))
        .await
        .unwrap()
        .into_inner();

    assert_eq!("frame-1", got.image_id);
}

#[tokio::test]
async fn list_returns_total_items() {
    let service = service();
    for _ in 0..3 {
        service
            .add(Request::new(AddRequest {
                image: solid_png(),
                bbox: Some(proto::BBox {
                    x: 0,
                    y: 0,
                    w: 4,
                    h: 4,
                }),
                image_id: String::new(),
                vehicle_id: String::new(),
                plate: String::new(),
            }))
            .await
            .unwrap();
    }

    let page = service
        .list(Request::new(ListRequest {
            page: 1,
            page_size: 2,
        }))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(3, page.total_items);
    assert_eq!(2, page.items.len());
}

#[tokio::test]
async fn set_plate_rejects_too_long_plate() {
    let service = service();
    let added = service
        .add(Request::new(AddRequest {
            image: solid_png(),
            bbox: Some(proto::BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            image_id: String::new(),
            vehicle_id: String::new(),
            plate: String::new(),
        }))
        .await
        .unwrap()
        .into_inner();

    let status = service
        .set_plate(Request::new(SetPlateRequest {
            id: added.id,
            plate: "x".repeat(33),
        }))
        .await
        .unwrap_err();

    assert_eq!(Code::InvalidArgument, status.code());
}

#[tokio::test]
async fn delete_returns_not_found_for_missing_id() {
    let service = service();

    let status = service
        .delete(Request::new(DeleteRequest { id: 404 }))
        .await
        .unwrap_err();

    assert_eq!(Code::NotFound, status.code());
}

#[tokio::test]
async fn set_tags_then_get_returns_the_tags() {
    let service = service();
    let added = service
        .add(Request::new(AddRequest {
            image: solid_png(),
            bbox: Some(proto::BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            image_id: String::new(),
            vehicle_id: String::new(),
            plate: String::new(),
        }))
        .await
        .unwrap()
        .into_inner();

    service
        .set_tags(Request::new(SetTagsRequest {
            item_id: added.id,
            tags: vec![proto::Tag {
                key: "roof_box".into(),
                confidence: 0.9,
                region: None,
            }],
        }))
        .await
        .unwrap();

    let got = service
        .get(Request::new(GetRequest { id: added.id }))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(1, got.tags.len());
    assert_eq!("roof_box", got.tags[0].key);
}
