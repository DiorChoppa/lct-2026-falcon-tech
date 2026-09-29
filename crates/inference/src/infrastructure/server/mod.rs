#[cfg(test)]
mod tests;

pub mod config;
pub mod constants;
mod responses;
mod routes;
mod swagger;

use anyhow::Context;
use axum::{routing::get, Router};
use tokio::{net::TcpListener, task::JoinHandle};
use tower_http::{cors, trace};

use super::setup::Application;

use constants::HEALTH_URL;

pub fn init_router() -> Router {
    let swagger_layer = swagger::init_api_doc();
    let cors_layer = cors::CorsLayer::permissive();
    let trace_layer = trace::TraceLayer::new_for_http()
        .make_span_with(trace::DefaultMakeSpan::new().level(tracing::Level::INFO))
        .on_response(trace::DefaultOnResponse::new().level(tracing::Level::INFO));

    Router::new()
        .merge(swagger_layer)
        .route(HEALTH_URL, get(routes::health))
        .layer(trace_layer)
        .layer(cors_layer)
}

pub async fn run_server(application: &Application) -> anyhow::Result<JoinHandle<()>> {
    let root_router = application.root_router.clone();
    let config = &application.config;
    let http_config = config.http_server();

    let listener = TcpListener::bind(http_config.address())
        .await
        .with_context(|| {
            format!(
                "Could not create TCP listener on: {}",
                http_config.address()
            )
        })?;

    let server_thread = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, root_router).await {
            tracing::error!(err=?err, "failed to stop http server");
        };
    });

    tracing::info!(
        "{} HTTP (health/swagger) started on address: http://{}.",
        config.system().application_name(),
        config.http_server().address()
    );

    Ok(server_thread)
}
