use std::path::PathBuf;

use config::{Config, ConfigError, Environment, File, FileFormat};
use dotenvy::dotenv;
use getset::{Getters, MutGetters};
use serde::Deserialize;

use crate::infrastructure::grpc::config::GrpcServerConfig;
use crate::infrastructure::server::config::HttpServerConfig;

use super::logging::LoggerConfig;

const SERVICE_RUN_MODE: &str = "INFERENCE__RUN_MODE";
const SERVICE_PREFIX: &str = "INFERENCE";

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct SystemConfig {
    application_name: String,
    production: bool,
}

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct ModelConfig {
    manifest_path: PathBuf,
}

/// `crops_dir` must be absolute — see infrastructure::grpc::crop_loader.
#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct StorageConfig {
    crops_dir: PathBuf,
}

#[derive(Deserialize, Getters, MutGetters)]
#[getset(get = "pub", get_mut = "pub")]
pub struct ServiceConfig {
    logger: LoggerConfig,
    system: SystemConfig,
    http_server: HttpServerConfig,
    grpc_server: GrpcServerConfig,
    model: ModelConfig,
    storage: StorageConfig,
}

impl ServiceConfig {
    pub fn new() -> Result<Self, ConfigError> {
        dotenv().ok();

        let run_mode = std::env::var(SERVICE_RUN_MODE).unwrap_or_else(|_| "development".into());
        // Workspace member, so CWD isn't the crate dir: look both next to the crate
        // manifest (local dev/tests) and relative to CWD (Docker's WORKDIR /app).
        let manifest_dir_path = format!("{}/config/{}", env!("CARGO_MANIFEST_DIR"), run_mode);
        let cwd_path = format!("./config/{}", run_mode);
        let manifest_dir_config = File::with_name(&manifest_dir_path)
            .format(FileFormat::Toml)
            .required(false);
        let cwd_config = File::with_name(&cwd_path)
            .format(FileFormat::Toml)
            .required(false);
        let env_config = Environment::with_prefix(SERVICE_PREFIX)
            .prefix_separator("__")
            .separator("__");
        let settings = Config::builder()
            .add_source(manifest_dir_config)
            .add_source(cwd_config)
            .add_source(env_config)
            .build()?;

        settings.try_deserialize()
    }
}
