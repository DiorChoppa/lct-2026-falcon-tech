pub mod application;
pub mod domain;
pub mod infrastructure;
#[cfg(test)]
pub mod test_utils;

use std::sync::Arc;

use anyhow::Context;
use inference_client::GrpcEmbeddingClient;
use sqlx::postgres::PgPoolOptions;
use storage::ObjectStoreRepository;
use tokio::sync::Notify;

use application::use_cases::UseCases;
use domain::ThresholdManifest;
use infrastructure::grpc;
use infrastructure::repositories::{SqlxGalleryReadRepository, SqlxSearchRepository};
use infrastructure::setup::{init_logger, Application, ServiceConfig};

pub async fn run(stop_signal: Arc<Notify>) -> anyhow::Result<()> {
    let config = ServiceConfig::new()?;
    init_logger(config.logger()).context("[logger] Failed to initialize logger")?;

    let application = Arc::new(init_app(config).await?);
    let grpc_thread = grpc::run_server(&application)
        .await
        .context("Can't create gRPC server thread")?;

    tokio::select! {
        _ = stop_signal.notified() => {
            tracing::info!("stop signal has been received");
        }
        res = grpc_thread => {
            tracing::warn!(result=?res, "grpc server task has been finished");
        }
    }
    Ok(())
}

pub async fn init_app(config: ServiceConfig) -> anyhow::Result<Application> {
    let pool = PgPoolOptions::new()
        .connect(config.database().url())
        .await
        .context("connecting to database")?;

    let threshold = ThresholdManifest::load(config.model().manifest_path())
        .context("loading models/model.json")?;
    let gallery_read = Arc::new(SqlxGalleryReadRepository::new(pool.clone()));
    let search_repository = Arc::new(SqlxSearchRepository::new(pool));
    let object_repository = Arc::new(ObjectStoreRepository::new(config.storage().bucket_url())?);
    let embedding_client = Arc::new(GrpcEmbeddingClient {
        url: config.inference().url().clone(),
    });
    let use_cases = Arc::new(UseCases::new(
        gallery_read,
        search_repository,
        object_repository,
        embedding_client,
        threshold,
    ));

    Ok(Application { config, use_cases })
}
