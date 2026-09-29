use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};
use proto::inference_server::Inference;
use proto::{BBox, CropRef, EmbedRequest, FrameWithBoxes};
use rstest::rstest;
use tonic::Request;
use url::Url;

use crate::application::use_cases::UseCases;
use crate::domain::ModelManifest;
use crate::infrastructure::grpc::crop_loader::CropLoader;
use crate::infrastructure::grpc::service::{convert_bbox, InferenceGrpcService};
use crate::infrastructure::repositories::OrtEmbeddingRepository;
use crate::test_utils::mock_embedding_repository::MockEmbeddingRepository;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn manifest() -> ModelManifest {
    ModelManifest {
        name: "test".into(),
        version: "0".into(),
        file: "smoke_model.onnx".into(),
        input_name: "input".into(),
        output_name: "embedding".into(),
        input_height: 4,
        input_width: 4,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
        resize: "bilinear".into(),
        dim: 3,
        l2_normalized: true,
        patches_output_name: None,
        patch_grid_h: None,
        patch_grid_w: None,
        patch_dim: None,
    }
}

fn no_crops_loader() -> CropLoader {
    CropLoader::new(PathBuf::from("/nonexistent"))
}

fn solid_png(rgb: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(4, 4, image::Rgb(rgb));
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&img, 4, 4, ExtendedColorType::Rgb8)
        .unwrap();
    out
}

#[rstest]
#[case::negative_x(BBox { x: -1, y: 0, w: 1, h: 1 })]
#[case::negative_y(BBox { x: 0, y: -1, w: 1, h: 1 })]
#[case::negative_w(BBox { x: 0, y: 0, w: -1, h: 1 })]
#[case::negative_h(BBox { x: 0, y: 0, w: 1, h: -1 })]
fn convert_bbox_rejects_negative_coordinates(#[case] bbox: BBox) {
    let err = convert_bbox(&bbox).unwrap_err();
    assert!(err.to_string().contains("negative coordinates"));
}

#[tokio::test]
async fn embed_maps_frames_and_boxes_to_response_in_order() {
    let repository = MockEmbeddingRepository::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
    let use_cases = UseCases::new(manifest(), Arc::new(repository));
    let service = InferenceGrpcService::new(Arc::new(use_cases), no_crops_loader());

    let frame = FrameWithBoxes {
        image: solid_png([51, 102, 204]),
        boxes: vec![
            BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            BBox {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
        ],
        frame_id: "frame-1".into(),
    };
    let request = Request::new(EmbedRequest {
        frames: vec![frame],
        crops: vec![],
        with_patches: false,
    });

    let response = service.embed(request).await.unwrap().into_inner();

    assert_eq!(2, response.embeddings.len());
    assert_eq!("frame-1", response.embeddings[0].frame_id);
    assert_eq!(0, response.embeddings[0].box_index);
    assert_eq!(vec![1.0, 0.0, 0.0], response.embeddings[0].values);
    assert_eq!("frame-1", response.embeddings[1].frame_id);
    assert_eq!(1, response.embeddings[1].box_index);
    assert_eq!(vec![0.0, 1.0, 0.0], response.embeddings[1].values);
    assert!(response.inference_us >= 0);
}

#[tokio::test]
async fn embed_rejects_negative_bbox_with_invalid_argument() {
    let repository = MockEmbeddingRepository::default();
    let use_cases = UseCases::new(manifest(), Arc::new(repository));
    let service = InferenceGrpcService::new(Arc::new(use_cases), no_crops_loader());

    let frame = FrameWithBoxes {
        image: solid_png([0, 0, 0]),
        boxes: vec![BBox {
            x: -1,
            y: 0,
            w: 1,
            h: 1,
        }],
        frame_id: "frame-1".into(),
    };
    let request = Request::new(EmbedRequest {
        frames: vec![frame],
        crops: vec![],
        with_patches: false,
    });

    let status = service.embed(request).await.unwrap_err();

    assert_eq!(tonic::Code::InvalidArgument, status.code());
}

#[tokio::test]
async fn embed_end_to_end_through_real_ort_smoke_model() {
    let manifest = manifest();
    let repository = OrtEmbeddingRepository::new(&manifest, &fixtures_dir());
    let use_cases = UseCases::new(manifest, Arc::new(repository));
    let service = InferenceGrpcService::new(Arc::new(use_cases), no_crops_loader());

    let frame = FrameWithBoxes {
        image: solid_png([51, 102, 204]),
        boxes: vec![BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        }],
        frame_id: "frame-1".into(),
    };
    let request = Request::new(EmbedRequest {
        frames: vec![frame],
        crops: vec![],
        with_patches: false,
    });

    let response = service.embed(request).await.unwrap().into_inner();

    assert_eq!(1, response.embeddings.len());
    assert_eq!(3, response.embeddings[0].values.len());
    assert!(response.embeddings[0].patches.is_empty());
}

#[tokio::test]
async fn embed_resolves_crop_uri_from_object_store() {
    let dir = tempfile::tempdir().unwrap();
    let crop_path = dir.path().join("abc.png");
    std::fs::write(&crop_path, solid_png([0, 255, 0])).unwrap();

    let repository = MockEmbeddingRepository::new(vec![vec![7.0, 8.0, 9.0]]);
    let use_cases = UseCases::new(manifest(), Arc::new(repository));
    let service = InferenceGrpcService::new(
        Arc::new(use_cases),
        CropLoader::new(dir.path().to_path_buf()),
    );

    let request = Request::new(EmbedRequest {
        frames: vec![],
        crops: vec![CropRef {
            uri: Url::from_file_path(&crop_path).unwrap().to_string(),
            frame_id: "frame-1".into(),
            box_index: 0,
        }],
        with_patches: false,
    });

    let response = service.embed(request).await.unwrap().into_inner();

    assert_eq!(1, response.embeddings.len());
    assert_eq!("frame-1", response.embeddings[0].frame_id);
    assert_eq!(vec![7.0, 8.0, 9.0], response.embeddings[0].values);
}

#[tokio::test]
async fn embed_rejects_crop_uri_escaping_configured_root() {
    let dir = tempfile::tempdir().unwrap();
    let repository = MockEmbeddingRepository::default();
    let use_cases = UseCases::new(manifest(), Arc::new(repository));
    let service = InferenceGrpcService::new(
        Arc::new(use_cases),
        CropLoader::new(dir.path().to_path_buf()),
    );

    // Url normalizes ".." during parsing, so this legitimately resolves
    // outside `dir` — exactly the case CropLoader must reject.
    let escaping = format!(
        "{}/../../etc/passwd",
        Url::from_file_path(dir.path()).unwrap()
    );
    let request = Request::new(EmbedRequest {
        frames: vec![],
        crops: vec![CropRef {
            uri: escaping,
            frame_id: "frame-1".into(),
            box_index: 0,
        }],
        with_patches: false,
    });

    let status = service.embed(request).await.unwrap_err();

    assert_eq!(tonic::Code::NotFound, status.code());
}

#[tokio::test]
async fn embed_rejects_with_patches_when_model_has_no_patches_output() {
    let repository = MockEmbeddingRepository::default();
    let use_cases = UseCases::new(manifest(), Arc::new(repository));
    let service = InferenceGrpcService::new(Arc::new(use_cases), no_crops_loader());

    let frame = FrameWithBoxes {
        image: solid_png([0, 0, 0]),
        boxes: vec![BBox {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        }],
        frame_id: "frame-1".into(),
    };
    let request = Request::new(EmbedRequest {
        frames: vec![frame],
        crops: vec![],
        with_patches: true,
    });

    let status = service.embed(request).await.unwrap_err();

    assert_eq!(tonic::Code::FailedPrecondition, status.code());
}

#[tokio::test]
async fn info_reports_no_patch_support_for_embedding_only_model() {
    let use_cases = UseCases::new(manifest(), Arc::new(MockEmbeddingRepository::default()));
    let service = InferenceGrpcService::new(Arc::new(use_cases), no_crops_loader());

    let response = service
        .info(Request::new(proto::InfoRequest {}))
        .await
        .unwrap()
        .into_inner();

    assert!(!response.supports_patches);
}

/// Holds the calling thread for the whole forward pass, like ORT on CPU.
struct BlockingRepository(std::time::Duration);

impl crate::application::repositories::EmbeddingRepository for BlockingRepository {
    fn embed(
        &self,
        batch: ndarray::Array4<f32>,
        _with_patches: bool,
    ) -> anyhow::Result<Vec<crate::application::repositories::EmbedOutput>> {
        std::thread::sleep(self.0);
        Ok(vec![
            crate::application::repositories::EmbedOutput {
                embedding: vec![1.0, 0.0, 0.0],
                patches: None,
            };
            batch.shape()[0]
        ])
    }
}

// Single-threaded runtime on purpose: an Embed that runs the model on the async
// worker would stall every other RPC (health, Info) until the forward pass ends.
#[tokio::test]
async fn info_answers_while_embed_runs_the_model() {
    let forward = std::time::Duration::from_millis(500);
    let use_cases = UseCases::new(manifest(), Arc::new(BlockingRepository(forward)));
    let service = Arc::new(InferenceGrpcService::new(
        Arc::new(use_cases),
        no_crops_loader(),
    ));
    let request = Request::new(EmbedRequest {
        frames: vec![FrameWithBoxes {
            image: solid_png([0, 0, 0]),
            boxes: vec![BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }],
            frame_id: "frame-1".into(),
        }],
        crops: vec![],
        with_patches: false,
    });

    let started = std::time::Instant::now();
    let embedding = tokio::spawn({
        let service = service.clone();
        async move { service.embed(request).await }
    });
    tokio::task::yield_now().await;
    service
        .info(Request::new(proto::InfoRequest {}))
        .await
        .unwrap();
    let info_after = started.elapsed();

    assert!(
        info_after < forward / 2,
        "Info waited {info_after:?} behind a running Embed"
    );
    assert_eq!(
        1,
        embedding
            .await
            .unwrap()
            .unwrap()
            .into_inner()
            .embeddings
            .len()
    );
}
