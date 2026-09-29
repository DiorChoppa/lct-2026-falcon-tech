pub mod config;
mod service;
#[cfg(test)]
mod tests;

use proto::search_server::SearchServer;
use tokio::task::JoinHandle;
use tonic::transport::Server;

use service::SearchGrpcService;

use super::setup::Application;

pub async fn run_server(application: &Application) -> anyhow::Result<JoinHandle<()>> {
    let grpc_config = application.config.grpc_server();
    let addr = grpc_config.address().parse()?;
    let use_cases = application.use_cases.clone();

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<SearchServer<SearchGrpcService>>()
        .await;

    let server_thread = tokio::spawn(async move {
        let service = SearchGrpcService::new(use_cases);
        if let Err(err) = Server::builder()
            .add_service(health_service)
            .add_service(SearchServer::new(service))
            .serve(addr)
            .await
        {
            tracing::error!(err=?err, "failed to stop grpc server");
        }
    });

    tracing::info!("search gRPC started on address: {}.", grpc_config.address());

    Ok(server_thread)
}
