#[cfg(test)]
mod tests;

use std::sync::Arc;

use common::BBox;

use crate::application::repositories::{
    EmbeddingClient, Frame, GalleryReadRepository, ObjectRepository, SearchRepository,
};
use crate::domain::{
    match_patches, patch_region, Candidate, NewSearch, PatchMatch, Region, SearchRecord, Tag,
    ThresholdManifest,
};

const CROP_JPEG_QUALITY: u8 = 95;

#[derive(Debug)]
pub struct RegionMatch {
    pub query_region: Region,
    pub candidate_region: Region,
    pub similarity: f32,
}

#[derive(Debug)]
pub struct CompareResult {
    pub matches: Vec<RegionMatch>,
    pub local_score: f32,
    pub note: String,
    pub candidate_tags: Vec<Tag>,
}

pub struct UseCases {
    gallery_read: Arc<dyn GalleryReadRepository>,
    search_repository: Arc<dyn SearchRepository>,
    object_repository: Arc<dyn ObjectRepository>,
    embedding_client: Arc<dyn EmbeddingClient>,
    threshold: ThresholdManifest,
}

impl UseCases {
    pub fn new(
        gallery_read: Arc<dyn GalleryReadRepository>,
        search_repository: Arc<dyn SearchRepository>,
        object_repository: Arc<dyn ObjectRepository>,
        embedding_client: Arc<dyn EmbeddingClient>,
        threshold: ThresholdManifest,
    ) -> Self {
        Self {
            gallery_read,
            search_repository,
            object_repository,
            embedding_client,
            threshold,
        }
    }

    pub async fn search(
        &self,
        image: Vec<u8>,
        bbox: BBox,
        top_n: u32,
        details: bool,
    ) -> anyhow::Result<SearchRecord> {
        let crop = common::crop_jpeg(&image, bbox, CROP_JPEG_QUALITY)?;
        let query_crop_uri = self
            .object_repository
            .put(&format!("{}.jpg", uuid::Uuid::new_v4()), crop)
            .await?;

        let embedded = self
            .embedding_client
            .embed(
                &[Frame {
                    image,
                    boxes: vec![bbox],
                }],
                false,
            )
            .await?;
        let embedding = embedded
            .into_iter()
            .next()
            .map(|e| e.embedding)
            .ok_or_else(|| anyhow::anyhow!("inference returned no embedding"))?;
        let info = self.embedding_client.info().await?;

        let rows = self
            .gallery_read
            .knn(&embedding, &info.model_version, top_n.max(1))
            .await?;

        let want_local =
            (details || self.threshold.alpha < 1.0) && info.supports_patches && !rows.is_empty();
        let mut candidates = Vec::with_capacity(rows.len());
        for row in rows {
            let (confidence, local_score) = if want_local {
                let local = self
                    .local_score(&query_crop_uri, &row.crop_uri, info.patch_dim)
                    .await?;
                let confidence = (self.threshold.alpha * row.score
                    + (1.0 - self.threshold.alpha) * local)
                    .clamp(0.0, 1.0);
                (confidence, Some(local))
            } else {
                (row.score, None)
            };
            candidates.push(Candidate {
                gallery_id: row.gallery_id,
                score: row.score,
                confidence,
                local_score,
                plate: row.plate,
                crop_uri: row.crop_uri,
            });
        }
        // Топ-N возвращается всегда, отказ — это флаг, а не пустой список: оператор
        // видит ближайших и ниже порога (web помечает их), жюри-артефакт не затронут.
        candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let accepted = candidates
            .iter()
            .any(|c| c.confidence >= self.threshold.threshold);

        self.search_repository
            .insert(NewSearch {
                query_crop_uri,
                bbox,
                candidates,
                accepted,
            })
            .await
    }

    async fn local_score(
        &self,
        query_uri: &str,
        candidate_uri: &str,
        patch_dim: u32,
    ) -> anyhow::Result<f32> {
        let (query_patches, candidate_patches) =
            self.patches_for_pair(query_uri, candidate_uri).await?;
        let (_, local_score) =
            match_patches(&query_patches, &candidate_patches, patch_dim as usize);
        Ok(local_score)
    }

    async fn patches_for_pair(
        &self,
        query_uri: &str,
        candidate_uri: &str,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let embedded = self
            .embedding_client
            .embed_by_uri(&[query_uri.to_string(), candidate_uri.to_string()], true)
            .await?;
        let mut iter = embedded.into_iter();
        let query = iter
            .next()
            .and_then(|e| e.patches)
            .ok_or_else(|| anyhow::anyhow!("inference returned no patches for {query_uri}"))?;
        let candidate = iter
            .next()
            .and_then(|e| e.patches)
            .ok_or_else(|| anyhow::anyhow!("inference returned no patches for {candidate_uri}"))?;
        Ok((query, candidate))
    }

    pub async fn compare(&self, search_id: i64, gallery_id: i64) -> anyhow::Result<CompareResult> {
        let search = self
            .search_repository
            .get(search_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("search {search_id} not found"))?;
        let candidate_uri = self
            .gallery_read
            .get_crop_uri(gallery_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("gallery item {gallery_id} not found"))?;

        let info = self.embedding_client.info().await?;
        anyhow::ensure!(info.supports_patches, "model has no patches output");

        let (query_patches, candidate_patches) = self
            .patches_for_pair(&search.query_crop_uri, &candidate_uri)
            .await?;
        let (matches, local_score) =
            match_patches(&query_patches, &candidate_patches, info.patch_dim as usize);
        let note = if matches.is_empty() {
            "few_shared_views".to_string()
        } else {
            String::new()
        };
        let region_matches = matches
            .into_iter()
            .map(|m: PatchMatch| RegionMatch {
                query_region: patch_region(
                    m.query_index,
                    info.patch_grid_h,
                    info.patch_grid_w,
                    info.input_width,
                    info.input_height,
                ),
                candidate_region: patch_region(
                    m.candidate_index,
                    info.patch_grid_h,
                    info.patch_grid_w,
                    info.input_width,
                    info.input_height,
                ),
                similarity: m.similarity,
            })
            .collect();
        let candidate_tags = self.gallery_read.get_tags(gallery_id).await?;

        Ok(CompareResult {
            matches: region_matches,
            local_score,
            note,
            candidate_tags,
        })
    }

    pub async fn export_csv(&self, search_id: i64) -> anyhow::Result<Vec<u8>> {
        let search = self
            .search_repository
            .get(search_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("search {search_id} not found"))?;

        let mut writer = csv::Writer::from_writer(vec![]);
        writer.write_record(["gallery_id", "score", "confidence", "plate"])?;
        for candidate in &search.candidates {
            writer.write_record([
                candidate.gallery_id.to_string(),
                candidate.score.to_string(),
                candidate.confidence.to_string(),
                candidate.plate.clone().unwrap_or_default(),
            ])?;
        }
        Ok(writer.into_inner()?)
    }
}
