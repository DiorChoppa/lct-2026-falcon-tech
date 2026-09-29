use config::{Config, ConfigError, Environment, File, FileFormat};
use dotenvy::dotenv;
use getset::Getters;
use serde::Deserialize;

use crate::infrastructure::grpc::config::GrpcServerConfig;

use super::logging::LoggerConfig;

const SERVICE_RUN_MODE: &str = "SEARCH__RUN_MODE";
const SERVICE_PREFIX: &str = "SEARCH";

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct SystemConfig {
    application_name: String,
}

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct DatabaseConfig {
    url: String,
}

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct InferenceConfig {
    url: String,
}

/// `bucket_url` is an object_store URL for the whole query-crops bucket.
#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct StorageConfig {
    bucket_url: String,
}

/// Path to `models/model.json` — search reads only `threshold`/`alpha` from
/// it (domain::ThresholdManifest); `model_version` comes from inference.Info.
#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct ModelConfig {
    manifest_path: std::path::PathBuf,
}

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct ServiceConfig {
    logger: LoggerConfig,
    system: SystemConfig,
    grpc_server: GrpcServerConfig,
    database: DatabaseConfig,
    inference: InferenceConfig,
    storage: StorageConfig,
    model: ModelConfig,
}

impl ServiceConfig {
    pub fn new() -> Result<Self, ConfigError> {
        dotenv().ok();

        let run_mode = std::env::var(SERVICE_RUN_MODE).unwrap_or_else(|_| "development".into());
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
