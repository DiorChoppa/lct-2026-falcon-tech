mod application;
mod domain;
pub mod infrastructure;
#[cfg(test)]
pub mod test_utils;

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use tokio::sync::Notify;

use application::repositories::ManifestRepository;
use infrastructure::repositories::FileManifestRepository;
use infrastructure::setup::{init_logger, Application, ServiceConfig};
use infrastructure::{grpc, server};

pub use application::repositories::EmbeddingRepository;
pub use application::use_cases::UseCases;
pub use domain::{Crop, ModelManifest};
pub use infrastructure::repositories::OrtEmbeddingRepository;

pub async fn run<Fut, Guard, Configurator>(
    stop_signal: Arc<Notify>,
    configurator: Configurator,
) -> anyhow::Result<()>
where
    Fut: Future<Output = anyhow::Result<Guard>>,
    Configurator: FnOnce(Arc<Application>) -> Fut,
{
    let application = Arc::new(init_app(ServiceConfig::new()?).await?);

    init_logger(application.config.logger()).context("[logger] Failed to initialize logger")?;
    let _configurator_guard = configurator(application.clone()).await?;

    let http_thread = server::run_server(&application)
        .await
        .context("Can't create HTTP server thread")?;
    let grpc_thread = grpc::run_server(&application)
        .await
        .context("Can't create gRPC server thread")?;

    tokio::select! {
        _ = stop_signal.notified() => {
            tracing::info!("stop signal has been received");
        }
        res = http_thread => {
            tracing::warn!(result=?res, "http server task has been finished");
        }
        res = grpc_thread => {
            tracing::warn!(result=?res, "grpc server task has been finished");
        }
    }

    Ok(())
}

pub async fn init_app(config: ServiceConfig) -> anyhow::Result<Application> {
    let manifest_repository = FileManifestRepository::new(config.model().manifest_path().clone());
    let manifest = manifest_repository.load()?;
    tracing::info!(model = %manifest.name, version = %manifest.version, "manifest loaded");

    let model_dir = config
        .model()
        .manifest_path()
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let embedding_repository = OrtEmbeddingRepository::new(&manifest, model_dir);
    let use_cases = UseCases::new(manifest, Arc::new(embedding_repository));

    Ok(Application {
        root_router: server::init_router(),
        use_cases: Arc::new(use_cases),
        config,
    })
}
