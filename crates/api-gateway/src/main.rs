//! Точка входа api-gateway: адреса gRPC-сервисов, порог из model.json, HTTP-сервер.

use std::path::PathBuf;
use std::sync::Arc;

use api_gateway::clients::{self, GrpcGallery, GrpcInference, GrpcSearch};
use api_gateway::openapi::ApiDoc;
use api_gateway::{app, AppState};
use clap::Parser;
use utoipa::OpenApi;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, env = "API_ADDR", default_value = "0.0.0.0:8080")]
    addr: String,
    #[arg(long, env = "GALLERY_URL", default_value = "http://localhost:50054")]
    gallery_url: String,
    #[arg(long, env = "SEARCH_URL", default_value = "http://localhost:50053")]
    search_url: String,
    #[arg(long, env = "INFERENCE_URL", default_value = "http://localhost:50051")]
    inference_url: String,
    /// Общий том кропов (тот же, куда пишут gallery/search), отдаётся как /files/crops.
    #[arg(long, env = "CROPS_DIR", default_value = "data/crops")]
    crops_dir: PathBuf,
    /// Корень датасета для импорта по CSV (кадры в <dataset>/images).
    #[arg(long, env = "DATASET_DIR", default_value = "dataset")]
    dataset_dir: PathBuf,
    /// Манифест модели; отсюда берётся только `threshold` для ответа /api/search.
    #[arg(long, env = "MODEL_MANIFEST", default_value = "models/model.json")]
    model_manifest: PathBuf,
    /// Напечатать OpenAPI в stdout и выйти (`just openapi`).
    #[arg(long)]
    print_openapi: bool,
}

#[derive(serde::Deserialize)]
struct Manifest {
    threshold: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.print_openapi {
        println!("{}", ApiDoc::openapi().to_pretty_json()?);
        return Ok(());
    }
    tracing_subscriber::fmt().with_env_filter("info").init();

    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&args.model_manifest)?)?;
    // Абсолютный путь: crop_uri от gallery/search — абсолютные file:// URI,
    // а маппинг в /files/crops сравнивает строки (crops.rs). Симлинки не
    // разрешаем: gallery/search пишут в URI свой bucket_url буквально, а на
    // macOS `/tmp` → `/private/tmp` — canonicalize сломал бы совпадение.
    let crops_dir = std::path::absolute(&args.crops_dir)?;
    if !crops_dir.is_dir() {
        tracing::warn!(path = %crops_dir.display(), "CROPS_DIR не каталог, /files/crops не будет работать");
    }
    let state = AppState {
        gallery: Arc::new(GrpcGallery {
            channel: clients::channel(&args.gallery_url)?,
        }),
        search: Arc::new(GrpcSearch {
            channel: clients::channel(&args.search_url)?,
        }),
        inference: Arc::new(GrpcInference {
            channel: clients::channel(&args.inference_url)?,
        }),
        crops_dir,
        dataset_dir: args.dataset_dir,
        threshold: manifest.threshold,
    };
    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, threshold = manifest.threshold, "api-gateway listening");
    axum::serve(listener, app(state)).await?;
    Ok(())
}
