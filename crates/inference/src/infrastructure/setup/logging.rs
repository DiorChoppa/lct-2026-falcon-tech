use anyhow::Context;
use getset::Getters;
use serde::Deserialize;

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Json,
    Pretty,
}

#[derive(Deserialize, Getters)]
#[getset(get = "pub")]
pub struct LoggerConfig {
    format: LogFormat,
}

pub fn init_logger(config: &LoggerConfig) -> anyhow::Result<()> {
    match config.format() {
        LogFormat::Json => json_logger()?,
        LogFormat::Pretty => pretty_logger()?,
    }

    Ok(())
}

fn json_logger() -> anyhow::Result<()> {
    let json = tracing_subscriber::fmt().json();
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .from_env()
        .with_context(|| "Failed to build env filter for JSON logger")?;

    json.with_env_filter(env_filter)
        .with_level(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .try_init()
        .ok();
    Ok(())
}

fn pretty_logger() -> anyhow::Result<()> {
    let pretty = tracing_subscriber::fmt().pretty();
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .from_env()
        .with_context(|| "Failed to build env filter for pretty logger")?;

    pretty.with_env_filter(env_filter).try_init().ok();
    Ok(())
}
