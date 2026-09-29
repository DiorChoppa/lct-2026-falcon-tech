use std::sync::Arc;

use common::BBox;
use proto::search_server::Search;
use proto::{
    CompareRequest, CompareResponse, ExportRequest, ExportResponse, SearchRequest, SearchResponse,
};
use tonic::{Request, Response, Status};

use crate::application::use_cases::{RegionMatch, UseCases};
use crate::domain::{Candidate as DomainCandidate, Region, SearchRecord, Tag};

pub struct SearchGrpcService {
    use_cases: Arc<UseCases>,
}

impl SearchGrpcService {
    pub fn new(use_cases: Arc<UseCases>) -> Self {
        Self { use_cases }
    }
}

fn convert_bbox(bbox: &proto::BBox) -> anyhow::Result<BBox> {
    anyhow::ensure!(
        bbox.x >= 0 && bbox.y >= 0 && bbox.w >= 0 && bbox.h >= 0,
        "bbox has negative coordinates: {bbox:?}"
    );
    Ok(BBox {
        x: bbox.x as u32,
        y: bbox.y as u32,
        w: bbox.w as u32,
        h: bbox.h as u32,
    })
}

fn to_proto_candidate(c: DomainCandidate) -> proto::Candidate {
    proto::Candidate {
        gallery_id: c.gallery_id,
        score: c.score,
        confidence: c.confidence,
        local_score: c.local_score.unwrap_or_default(),
        plate: c.plate.unwrap_or_default(),
        crop_uri: c.crop_uri,
    }
}

fn to_proto_response(record: SearchRecord) -> SearchResponse {
    SearchResponse {
        search_id: record.id,
        query_crop_uri: record.query_crop_uri,
        candidates: record
            .candidates
            .into_iter()
            .map(to_proto_candidate)
            .collect(),
        accepted: record.accepted,
    }
}

fn to_proto_region(region: Region) -> proto::Region {
    proto::Region {
        x: region.x as i32,
        y: region.y as i32,
        w: region.w as i32,
        h: region.h as i32,
    }
}

fn to_proto_match(m: RegionMatch) -> proto::PatchMatch {
    proto::PatchMatch {
        query_region: Some(to_proto_region(m.query_region)),
        candidate_region: Some(to_proto_region(m.candidate_region)),
        similarity: m.similarity,
    }
}

/// use_cases surfaces "not found" as a plain anyhow message (no typed
/// Option at this boundary, unlike gallery's repository trait) — matching
/// on the message is a pragmatic compromise over a bigger error-type
/// refactor, not a claim that this is the most robust classification.
fn to_status(err: anyhow::Error) -> Status {
    if err.to_string().contains("not found") {
        Status::not_found(err.to_string())
    } else {
        Status::internal(err.to_string())
    }
}

fn to_proto_tag(tag: Tag) -> proto::Tag {
    proto::Tag {
        key: tag.key,
        confidence: tag.confidence,
        region: tag.region.map(|r| proto::Region {
            x: r.x as i32,
            y: r.y as i32,
            w: r.w as i32,
            h: r.h as i32,
        }),
    }
}

#[tonic::async_trait]
impl Search for SearchGrpcService {
    async fn search(
        &self,
        req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = req.into_inner();
        let bbox = convert_bbox(
            req.bbox
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("bbox is required"))?,
        )
        .map_err(|err| Status::invalid_argument(err.to_string()))?;

        let record = self
            .use_cases
            .search(req.image, bbox, req.top_n.max(1), req.details)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(to_proto_response(record)))
    }

    async fn compare(
        &self,
        req: Request<CompareRequest>,
    ) -> Result<Response<CompareResponse>, Status> {
        let req = req.into_inner();
        let result = self
            .use_cases
            .compare(req.search_id, req.gallery_id)
            .await
            .map_err(to_status)?;
        Ok(Response::new(CompareResponse {
            matches: result.matches.into_iter().map(to_proto_match).collect(),
            local_score: result.local_score,
            note: result.note,
            candidate_tags: result
                .candidate_tags
                .into_iter()
                .map(to_proto_tag)
                .collect(),
        }))
    }

    async fn export(
        &self,
        req: Request<ExportRequest>,
    ) -> Result<Response<ExportResponse>, Status> {
        let csv = self
            .use_cases
            .export_csv(req.into_inner().search_id)
            .await
            .map_err(to_status)?;
        Ok(Response::new(ExportResponse { csv }))
    }
}
