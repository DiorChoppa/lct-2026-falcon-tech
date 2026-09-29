use std::sync::Arc;

use common::BBox;
use proto::gallery_server::Gallery;
use proto::{
    AddRequest, DeleteRequest, DeleteResponse, GetRequest, ImportRequest, ImportResponse,
    ListRequest, ListResponse, ReindexRequest, ReindexResponse, SetPlateRequest, SetTagsRequest,
    SetTagsResponse,
};
use tonic::{Request, Response, Status};

use crate::application::structures::ImportItem;
use crate::application::use_cases::UseCases;
use crate::domain::{GalleryItem, Tag};

pub struct GalleryGrpcService {
    use_cases: Arc<UseCases>,
}

impl GalleryGrpcService {
    pub fn new(use_cases: Arc<UseCases>) -> Self {
        Self { use_cases }
    }
}

pub(super) fn convert_bbox(bbox: &proto::BBox) -> anyhow::Result<BBox> {
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

fn to_proto_bbox(bbox: BBox) -> proto::BBox {
    proto::BBox {
        x: bbox.x as i32,
        y: bbox.y as i32,
        w: bbox.w as i32,
        h: bbox.h as i32,
    }
}

fn to_proto_region(bbox: BBox) -> proto::Region {
    proto::Region {
        x: bbox.x as i32,
        y: bbox.y as i32,
        w: bbox.w as i32,
        h: bbox.h as i32,
    }
}

fn from_proto_region(region: &proto::Region) -> anyhow::Result<BBox> {
    anyhow::ensure!(
        region.x >= 0 && region.y >= 0 && region.w >= 0 && region.h >= 0,
        "tag region has negative coordinates: {region:?}"
    );
    Ok(BBox {
        x: region.x as u32,
        y: region.y as u32,
        w: region.w as u32,
        h: region.h as u32,
    })
}

fn to_proto_tag(tag: Tag) -> proto::Tag {
    proto::Tag {
        key: tag.key,
        confidence: tag.confidence,
        region: tag.region.map(to_proto_region),
    }
}

fn from_proto_tag(tag: proto::Tag) -> anyhow::Result<Tag> {
    let region = tag.region.as_ref().map(from_proto_region).transpose()?;
    Ok(Tag {
        key: tag.key,
        confidence: tag.confidence,
        region,
    })
}

fn to_proto_item(item: GalleryItem) -> proto::GalleryItem {
    proto::GalleryItem {
        id: item.id,
        image_id: item.image_id,
        vehicle_id: item.vehicle_id.unwrap_or_default(),
        bbox: Some(to_proto_bbox(item.bbox)),
        crop_uri: item.crop_uri,
        plate: item.plate.unwrap_or_default(),
        tags: item.tags.into_iter().map(to_proto_tag).collect(),
        model_version: item.model_version,
        created_at_unix: item.created_at.timestamp(),
    }
}

fn not_found() -> Status {
    Status::not_found("gallery item not found")
}

#[tonic::async_trait]
impl Gallery for GalleryGrpcService {
    async fn add(&self, req: Request<AddRequest>) -> Result<Response<proto::GalleryItem>, Status> {
        let req = req.into_inner();
        let bbox = convert_bbox(
            req.bbox
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("bbox is required"))?,
        )
        .map_err(|err| Status::invalid_argument(err.to_string()))?;

        let item = self
            .use_cases
            .add(
                req.image,
                bbox,
                non_empty(req.image_id),
                non_empty(req.vehicle_id),
                non_empty(req.plate),
            )
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(to_proto_item(item)))
    }

    async fn import(
        &self,
        req: Request<ImportRequest>,
    ) -> Result<Response<ImportResponse>, Status> {
        let req = req.into_inner();
        let mut items = Vec::with_capacity(req.items.len());
        for item in req.items {
            let bbox = convert_bbox(
                item.bbox
                    .as_ref()
                    .ok_or_else(|| Status::invalid_argument("bbox is required"))?,
            )
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
            items.push(ImportItem {
                image: item.image,
                bbox,
                image_id: item.image_id,
                vehicle_id: non_empty(item.vehicle_id),
            });
        }

        let report = self
            .use_cases
            .import(items)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(ImportResponse {
            imported: report.imported as i32,
            failed: report.failed as i32,
            errors: report
                .errors
                .into_iter()
                .map(|(image_id, message)| proto::ImportError { image_id, message })
                .collect(),
        }))
    }

    async fn get(&self, req: Request<GetRequest>) -> Result<Response<proto::GalleryItem>, Status> {
        let item = self
            .use_cases
            .get(req.into_inner().id)
            .await
            .map_err(|err| Status::internal(err.to_string()))?
            .ok_or_else(not_found)?;
        Ok(Response::new(to_proto_item(item)))
    }

    async fn list(&self, req: Request<ListRequest>) -> Result<Response<ListResponse>, Status> {
        let req = req.into_inner();
        let page = self
            .use_cases
            .list(req.page, req.page_size)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(ListResponse {
            items: page.items.into_iter().map(to_proto_item).collect(),
            total_items: page.total_items,
        }))
    }

    async fn set_plate(
        &self,
        req: Request<SetPlateRequest>,
    ) -> Result<Response<proto::GalleryItem>, Status> {
        let req = req.into_inner();
        let item = self
            .use_cases
            .set_plate(req.id, non_empty(req.plate))
            .await
            .map_err(|err| Status::invalid_argument(err.to_string()))?
            .ok_or_else(not_found)?;
        Ok(Response::new(to_proto_item(item)))
    }

    async fn set_tags(
        &self,
        req: Request<SetTagsRequest>,
    ) -> Result<Response<SetTagsResponse>, Status> {
        let req = req.into_inner();
        let tags = req
            .tags
            .into_iter()
            .map(from_proto_tag)
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        self.use_cases
            .set_tags(req.item_id, tags)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(SetTagsResponse {}))
    }

    async fn delete(
        &self,
        req: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let deleted = self
            .use_cases
            .delete(req.into_inner().id)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        if !deleted {
            return Err(not_found());
        }
        Ok(Response::new(DeleteResponse {}))
    }

    async fn reindex(
        &self,
        req: Request<ReindexRequest>,
    ) -> Result<Response<ReindexResponse>, Status> {
        let reindexed = self
            .use_cases
            .reindex(non_empty(req.into_inner().model_version))
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(ReindexResponse {
            reindexed: reindexed as i32,
        }))
    }
}

fn non_empty(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
