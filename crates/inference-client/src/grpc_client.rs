use std::collections::HashMap;

use proto::inference_client::InferenceClient;
use proto::{CropRef, EmbedRequest, FrameWithBoxes, InfoRequest};

use crate::{Embedded, EmbeddingClient, Frame, ModelInfo};

pub struct GrpcEmbeddingClient {
    pub url: String,
}

impl GrpcEmbeddingClient {
    async fn client(&self) -> anyhow::Result<InferenceClient<tonic::transport::Channel>> {
        InferenceClient::connect(self.url.clone())
            .await
            .map(|c| {
                c.max_encoding_message_size(64 << 20)
                    .max_decoding_message_size(64 << 20)
            })
            .map_err(|err| anyhow::anyhow!("inference unavailable at {}: {err}", self.url))
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for GrpcEmbeddingClient {
    async fn embed(&self, frames: &[Frame], with_patches: bool) -> anyhow::Result<Vec<Embedded>> {
        let request = EmbedRequest {
            frames: frames
                .iter()
                .enumerate()
                .map(|(i, f)| FrameWithBoxes {
                    image: f.image.clone(),
                    boxes: f
                        .boxes
                        .iter()
                        .map(|b| proto::BBox {
                            x: b.x as i32,
                            y: b.y as i32,
                            w: b.w as i32,
                            h: b.h as i32,
                        })
                        .collect(),
                    frame_id: i.to_string(),
                })
                .collect(),
            crops: vec![],
            with_patches,
        };
        let expected: usize = frames.iter().map(|f| f.boxes.len()).sum();

        let resp = self
            .client()
            .await?
            .embed(request)
            .await
            .map_err(|status| anyhow::anyhow!("inference Embed failed: {status}"))?
            .into_inner();
        anyhow::ensure!(
            resp.embeddings.len() == expected,
            "expected {expected} embeddings from inference, got {}",
            resp.embeddings.len()
        );
        Ok(resp
            .embeddings
            .into_iter()
            .map(|e| Embedded {
                embedding: e.values,
                patches: (!e.patches.is_empty()).then_some(e.patches),
            })
            .collect())
    }

    async fn embed_by_uri(
        &self,
        uris: &[String],
        with_patches: bool,
    ) -> anyhow::Result<Vec<Embedded>> {
        if uris.is_empty() {
            return Ok(vec![]);
        }
        let request = EmbedRequest {
            frames: vec![],
            crops: uris
                .iter()
                .map(|uri| CropRef {
                    uri: uri.clone(),
                    // frame_id round-trips the uri itself so the response can be
                    // matched back to `uris` order regardless of what inference
                    // preserves internally.
                    frame_id: uri.clone(),
                    box_index: 0,
                })
                .collect(),
            with_patches,
        };

        let resp = self
            .client()
            .await?
            .embed(request)
            .await
            .map_err(|status| anyhow::anyhow!("inference Embed failed: {status}"))?
            .into_inner();
        anyhow::ensure!(
            resp.embeddings.len() == uris.len(),
            "expected {} embeddings from inference, got {}",
            uris.len(),
            resp.embeddings.len()
        );

        let mut by_uri: HashMap<String, Embedded> = resp
            .embeddings
            .into_iter()
            .map(|e| {
                (
                    e.frame_id,
                    Embedded {
                        embedding: e.values,
                        patches: (!e.patches.is_empty()).then_some(e.patches),
                    },
                )
            })
            .collect();
        uris.iter()
            .map(|uri| {
                by_uri
                    .remove(uri)
                    .ok_or_else(|| anyhow::anyhow!("inference response missing crop_uri {uri}"))
            })
            .collect()
    }

    async fn info(&self) -> anyhow::Result<ModelInfo> {
        let resp = self
            .client()
            .await?
            .info(InfoRequest {})
            .await
            .map_err(|status| anyhow::anyhow!("inference Info failed: {status}"))?
            .into_inner();
        Ok(ModelInfo {
            model_version: resp.model_version,
            dim: resp.dim as u32,
            input_height: resp.input_height as u32,
            input_width: resp.input_width as u32,
            supports_patches: resp.supports_patches,
            patch_grid_h: resp.patch_grid_h as u32,
            patch_grid_w: resp.patch_grid_w as u32,
            patch_dim: resp.patch_dim as u32,
        })
    }
}
