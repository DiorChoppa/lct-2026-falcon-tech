#[cfg(test)]
mod tests;

pub mod config;
mod crop_loader;
mod service;

use proto::inference_server::InferenceServer;
use tokio::task::JoinHandle;
use tonic::transport::Server;

use crop_loader::CropLoader;
use service::InferenceGrpcService;

use super::setup::Application;

pub async fn run_server(application: &Application) -> anyhow::Result<JoinHandle<()>> {
    let config = &application.config;
    let grpc_config = config.grpc_server();
    let addr = grpc_config.address().parse()?;
    let use_cases = application.use_cases.clone();
    let crop_loader = CropLoader::new(config.storage().crops_dir().clone());

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<InferenceServer<InferenceGrpcService>>()
        .await;

    let server_thread = tokio::spawn(async move {
        let service = InferenceGrpcService::new(use_cases, crop_loader);
        if let Err(err) = Server::builder()
            .add_service(health_service)
            .add_service(InferenceServer::new(service))
            .serve(addr)
            .await
        {
            tracing::error!(err=?err, "failed to stop grpc server");
        }
    });

    tracing::info!(
        "{} gRPC started on address: {}.",
        config.system().application_name(),
        grpc_config.address()
    );

    Ok(server_thread)
}
