pub mod application;
pub mod domain;
pub mod infrastructure;
#[cfg(test)]
pub mod test_utils;

use std::sync::Arc;

use anyhow::Context;
use sqlx::postgres::PgPoolOptions;
use storage::ObjectStoreRepository;
use tokio::sync::Notify;

use application::use_cases::UseCases;
use inference_client::GrpcEmbeddingClient;
use infrastructure::grpc;
use infrastructure::repositories::{GrpcTaggerClient, SqlxGalleryRepository};
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
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .context("running migrations")?;

    let gallery_repository = Arc::new(SqlxGalleryRepository::new(pool));
    let object_repository = Arc::new(ObjectStoreRepository::new(config.storage().bucket_url())?);
    let embedding_client = Arc::new(GrpcEmbeddingClient {
        url: config.inference().url().clone(),
    });
    let tagger_client = Arc::new(GrpcTaggerClient {
        url: config.tagger().url().clone(),
    });
    let use_cases = Arc::new(UseCases::new(
        gallery_repository,
        object_repository,
        embedding_client,
        tagger_client,
    ));

    Ok(Application { config, use_cases })
}
