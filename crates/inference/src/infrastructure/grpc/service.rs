use std::sync::Arc;
use std::time::Instant;

use proto::inference_server::Inference;
use proto::{EmbedRequest, EmbedResponse, Embedding, InfoRequest, InfoResponse};
use tonic::{Request, Response, Status};

use super::crop_loader::CropLoader;
use crate::application::use_cases::UseCases;
use crate::domain::Crop;

pub struct InferenceGrpcService {
    use_cases: Arc<UseCases>,
    crop_loader: CropLoader,
}

impl InferenceGrpcService {
    pub fn new(use_cases: Arc<UseCases>, crop_loader: CropLoader) -> Self {
        Self {
            use_cases,
            crop_loader,
        }
    }
}

/// proto::BBox uses int32 (wire format), common::BBox uses u32 (dataset contract).
pub(super) fn convert_bbox(bbox: &proto::BBox) -> anyhow::Result<common::BBox> {
    anyhow::ensure!(
        bbox.x >= 0 && bbox.y >= 0 && bbox.w >= 0 && bbox.h >= 0,
        "bbox has negative coordinates: {bbox:?}"
    );
    Ok(common::BBox {
        x: bbox.x as u32,
        y: bbox.y as u32,
        w: bbox.w as u32,
        h: bbox.h as u32,
    })
}

/// crop_uri already points at a pre-cropped image, so the bbox is "the whole
/// frame" — `common::BBox::clamp` trims this down to the real dimensions once
/// the image is decoded.
const FULL_IMAGE_BBOX: common::BBox = common::BBox {
    x: 0,
    y: 0,
    w: u32::MAX,
    h: u32::MAX,
};

#[tonic::async_trait]
impl Inference for InferenceGrpcService {
    async fn embed(&self, req: Request<EmbedRequest>) -> Result<Response<EmbedResponse>, Status> {
        let req = req.into_inner();

        let mut crops = Vec::new();
        let mut keys = Vec::new();
        for frame in &req.frames {
            let image: Arc<[u8]> = Arc::from(frame.image.as_slice());
            for (box_index, b) in frame.boxes.iter().enumerate() {
                let bbox =
                    convert_bbox(b).map_err(|err| Status::invalid_argument(err.to_string()))?;
                crops.push(Crop {
                    image: image.clone(),
                    bbox,
                });
                keys.push((frame.frame_id.clone(), box_index as i32));
            }
        }
        for crop_ref in &req.crops {
            let bytes = self
                .crop_loader
                .load(&crop_ref.uri)
                .await
                .map_err(|err| Status::not_found(err.to_string()))?;
            crops.push(Crop {
                image: Arc::from(bytes),
                bbox: FULL_IMAGE_BBOX,
            });
            keys.push((crop_ref.frame_id.clone(), crop_ref.box_index));
        }

        if req.with_patches && !self.use_cases.info().supports_patches {
            return Err(Status::failed_precondition(
                "model has no patches output; check Info.supports_patches first",
            ));
        }

        // The forward pass is synchronous and takes seconds on CPU: run it off the
        // async workers so health checks and Info keep answering under load.
        let use_cases = self.use_cases.clone();
        let with_patches = req.with_patches;
        let started = Instant::now();
        let outputs = tokio::task::spawn_blocking(move || use_cases.embed(&crops, with_patches))
            .await
            .map_err(|err| Status::internal(err.to_string()))?
            .map_err(|err| Status::internal(err.to_string()))?;
        let inference_us = started.elapsed().as_micros() as i64;

        let embeddings = keys
            .into_iter()
            .zip(outputs)
            .map(|((frame_id, box_index), output)| Embedding {
                frame_id,
                box_index,
                values: output.embedding,
                patches: output.patches.unwrap_or_default(),
            })
            .collect();

        Ok(Response::new(EmbedResponse {
            embeddings,
            inference_us,
        }))
    }

    async fn info(&self, _req: Request<InfoRequest>) -> Result<Response<InfoResponse>, Status> {
        let info = self.use_cases.info();
        Ok(Response::new(InfoResponse {
            model_name: info.model_name,
            model_version: info.model_version,
            dim: info.dim as i32,
            input_height: info.input_height as i32,
            input_width: info.input_width as i32,
            execution_provider: info.execution_provider,
            supports_patches: info.supports_patches,
            patch_grid_h: info.patch_grid_h as i32,
            patch_grid_w: info.patch_grid_w as i32,
            patch_dim: info.patch_dim as i32,
        }))
    }
}
