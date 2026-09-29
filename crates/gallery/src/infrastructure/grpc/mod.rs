pub mod config;
mod service;
#[cfg(test)]
mod tests;

use proto::gallery_server::GalleryServer;
use tokio::task::JoinHandle;
use tonic::transport::Server;

use service::GalleryGrpcService;

use super::setup::Application;

pub async fn run_server(application: &Application) -> anyhow::Result<JoinHandle<()>> {
    let grpc_config = application.config.grpc_server();
    let addr = grpc_config.address().parse()?;
    let use_cases = application.use_cases.clone();

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<GalleryServer<GalleryGrpcService>>()
        .await;

    let server_thread = tokio::spawn(async move {
        let service = GalleryGrpcService::new(use_cases);
        if let Err(err) = Server::builder()
            .add_service(health_service)
            .add_service(GalleryServer::new(service))
            .serve(addr)
            .await
        {
            tracing::error!(err=?err, "failed to stop grpc server");
        }
    });

    tracing::info!(
        "gallery gRPC started on address: {}.",
        grpc_config.address()
    );

    Ok(server_thread)
}
