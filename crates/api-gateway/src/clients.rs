//! Клиенты gRPC к gallery, search и inference за трейтами: в тестах — фейки в
//! памяти. На сервис один ленивый `Channel` (создаётся в main, подключается
//! при первом вызове, сам переподключается после обрыва и мультиплексирует
//! параллельные вызовы — например, `gallery.Get` по всем кандидатам поиска —
//! в одно HTTP/2-соединение). Таймауты — из `04-architecture.md` §1. Ошибку
//! соединения tonic отдаёт как `Status::unavailable`, поэтому у обработчиков
//! один тип ошибки — `tonic::Status` — и одна точка маппинга в HTTP.

use std::time::Duration;

use async_trait::async_trait;
use proto::gallery_client::GalleryClient as GalleryStub;
use proto::inference_client::InferenceClient as InferenceStub;
use proto::search_client::SearchClient as SearchStub;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(1);
const GALLERY_TIMEOUT: Duration = Duration::from_secs(30);
const SEARCH_TIMEOUT: Duration = Duration::from_secs(10);
const INFERENCE_TIMEOUT: Duration = Duration::from_secs(5);
/// 64 МБ — как у inference-client. На приём это нужно для ListResponse и
/// ImportResponse; на отправку фактическая граница — 4 МБ у серверов
/// gallery/search/inference (дефолт tonic), поэтому кадр ограничен
/// `gallery::MAX_FRAME_BYTES`, а Import режется на пачки такого же размера.
const MAX_MESSAGE: usize = 64 << 20;

#[async_trait]
pub trait GalleryClient: Send + Sync {
    async fn add(&self, req: proto::AddRequest) -> Result<proto::GalleryItem, Status>;
    async fn import(&self, items: Vec<proto::ImportItem>) -> Result<proto::ImportResponse, Status>;
    async fn get(&self, id: i64) -> Result<proto::GalleryItem, Status>;
    async fn list(&self, page: u32, page_size: u32) -> Result<proto::ListResponse, Status>;
    /// Пустой `plate` снимает ГРЗ.
    async fn set_plate(&self, id: i64, plate: String) -> Result<proto::GalleryItem, Status>;
    async fn health(&self) -> Result<(), Status>;
}

#[async_trait]
pub trait SearchClient: Send + Sync {
    async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status>;
    /// Патч-токены query и кандидата из уже сохранённого поиска.
    async fn compare(
        &self,
        search_id: i64,
        gallery_id: i64,
    ) -> Result<proto::CompareResponse, Status>;
    /// CSV поиска в формате search (`gallery_id,score,confidence,plate`).
    async fn export(&self, search_id: i64) -> Result<Vec<u8>, Status>;
    async fn health(&self) -> Result<(), Status>;
}

#[async_trait]
pub trait InferenceClient: Send + Sync {
    async fn embed(&self, req: proto::EmbedRequest) -> Result<proto::EmbedResponse, Status>;
    async fn info(&self) -> Result<proto::InfoResponse, Status>;
    async fn health(&self) -> Result<(), Status>;
}

/// Ленивый канал: ошибка здесь — только некорректный URL.
pub fn channel(url: &str) -> Result<Channel, tonic::transport::Error> {
    Ok(Endpoint::from_shared(url.to_string())?
        .connect_timeout(CONNECT_TIMEOUT)
        .connect_lazy())
}

fn with_timeout<T>(msg: T, timeout: Duration) -> Request<T> {
    let mut req = Request::new(msg);
    req.set_timeout(timeout);
    req
}

/// grpc.health.v1.Health/Check по полному имени сервиса (`reid.v1.Gallery` и т. д.).
async fn check_health(channel: &Channel, service: &str) -> Result<(), Status> {
    let resp = HealthClient::new(channel.clone())
        .check(with_timeout(
            HealthCheckRequest {
                service: service.into(),
            },
            HEALTH_TIMEOUT,
        ))
        .await?
        .into_inner();
    if resp.status == ServingStatus::Serving as i32 {
        Ok(())
    } else {
        Err(Status::unavailable(format!(
            "{service}: health status {}",
            resp.status
        )))
    }
}

pub struct GrpcGallery {
    pub channel: Channel,
}

impl GrpcGallery {
    fn stub(&self) -> GalleryStub<Channel> {
        GalleryStub::new(self.channel.clone())
            .max_encoding_message_size(MAX_MESSAGE)
            .max_decoding_message_size(MAX_MESSAGE)
    }
}

#[async_trait]
impl GalleryClient for GrpcGallery {
    async fn add(&self, req: proto::AddRequest) -> Result<proto::GalleryItem, Status> {
        let resp = self.stub().add(with_timeout(req, GALLERY_TIMEOUT)).await?;
        Ok(resp.into_inner())
    }

    async fn import(&self, items: Vec<proto::ImportItem>) -> Result<proto::ImportResponse, Status> {
        let resp = self
            .stub()
            .import(with_timeout(
                proto::ImportRequest { items },
                GALLERY_TIMEOUT,
            ))
            .await?;
        Ok(resp.into_inner())
    }

    async fn get(&self, id: i64) -> Result<proto::GalleryItem, Status> {
        let resp = self
            .stub()
            .get(with_timeout(proto::GetRequest { id }, GALLERY_TIMEOUT))
            .await?;
        Ok(resp.into_inner())
    }

    async fn list(&self, page: u32, page_size: u32) -> Result<proto::ListResponse, Status> {
        let resp = self
            .stub()
            .list(with_timeout(
                proto::ListRequest { page, page_size },
                GALLERY_TIMEOUT,
            ))
            .await?;
        Ok(resp.into_inner())
    }

    async fn set_plate(&self, id: i64, plate: String) -> Result<proto::GalleryItem, Status> {
        let resp = self
            .stub()
            .set_plate(with_timeout(
                proto::SetPlateRequest { id, plate },
                GALLERY_TIMEOUT,
            ))
            .await?;
        Ok(resp.into_inner())
    }

    async fn health(&self) -> Result<(), Status> {
        check_health(&self.channel, "reid.v1.Gallery").await
    }
}

pub struct GrpcSearch {
    pub channel: Channel,
}

impl GrpcSearch {
    fn stub(&self) -> SearchStub<Channel> {
        SearchStub::new(self.channel.clone())
            .max_encoding_message_size(MAX_MESSAGE)
            .max_decoding_message_size(MAX_MESSAGE)
    }
}

#[async_trait]
impl SearchClient for GrpcSearch {
    async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
        let resp = self
            .stub()
            .search(with_timeout(req, SEARCH_TIMEOUT))
            .await?;
        Ok(resp.into_inner())
    }

    async fn compare(
        &self,
        search_id: i64,
        gallery_id: i64,
    ) -> Result<proto::CompareResponse, Status> {
        let resp = self
            .stub()
            .compare(with_timeout(
                proto::CompareRequest {
                    search_id,
                    gallery_id,
                },
                SEARCH_TIMEOUT,
            ))
            .await?;
        Ok(resp.into_inner())
    }

    async fn export(&self, search_id: i64) -> Result<Vec<u8>, Status> {
        let resp = self
            .stub()
            .export(with_timeout(
                proto::ExportRequest { search_id },
                SEARCH_TIMEOUT,
            ))
            .await?;
        Ok(resp.into_inner().csv)
    }

    async fn health(&self) -> Result<(), Status> {
        check_health(&self.channel, "reid.v1.Search").await
    }
}

pub struct GrpcInference {
    pub channel: Channel,
}

impl GrpcInference {
    fn stub(&self) -> InferenceStub<Channel> {
        InferenceStub::new(self.channel.clone())
            .max_encoding_message_size(MAX_MESSAGE)
            .max_decoding_message_size(MAX_MESSAGE)
    }
}

#[async_trait]
impl InferenceClient for GrpcInference {
    async fn embed(&self, req: proto::EmbedRequest) -> Result<proto::EmbedResponse, Status> {
        let resp = self
            .stub()
            .embed(with_timeout(req, INFERENCE_TIMEOUT))
            .await?;
        Ok(resp.into_inner())
    }

    async fn info(&self) -> Result<proto::InfoResponse, Status> {
        let resp = self
            .stub()
            .info(with_timeout(proto::InfoRequest {}, INFERENCE_TIMEOUT))
            .await?;
        Ok(resp.into_inner())
    }

    async fn health(&self) -> Result<(), Status> {
        check_health(&self.channel, "reid.v1.Inference").await
    }
}
