use anyhow::{Context, Result, ensure};
use lct_inference::{Config, Embeddings, Extractor, PreprocessingOptions};
use lct_offline::{role_batches, inputs, read_rows, resolve, sha256_file, write_submission};
use serde::Deserialize;
use std::{env, fs::File, path::{Path, PathBuf}};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinnedFile { path: PathBuf, sha256: String }

impl PinnedFile {
    fn verified_path(&self, base: &Path) -> Result<PathBuf> {
        ensure!(self.sha256.len() == 64 && self.sha256.bytes().all(|c| c.is_ascii_hexdigit()), "expected SHA256 must be64 hexadecimal characters");
        let path = resolve(base, &self.path);
        ensure!(sha256_file(&path)?.eq_ignore_ascii_case(&self.sha256), "checksum mismatch: {}", path.display());
        Ok(path)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model { graph: PinnedFile, preprocessing: Config, metadata_policy: Option<PinnedFile> }

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ThresholdStatus { DevelopmentOnly, FrozenCalibration }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DriverConfig {
    schema_version: u32,
    query_csv: PathBuf,
    gallery_csv: PathBuf,
    images_dir: PathBuf,
    runtime: PinnedFile,
    models: Vec<Model>,
    first_weight: Option<f32>,
    threshold: f64,
    threshold_status: ThresholdStatus,
    batch_size: usize,
    preprocessing_options: PreprocessingOptions,
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    ensure!(args.len() == 2, "usage: lct-offline CONFIG.json FRESH_OUTPUT_DIRECTORY");
    let config_path = PathBuf::from(&args[0]).canonicalize()?;
    let output = PathBuf::from(&args[1]);
    ensure!(!output.exists(), "refusing to overwrite output directory");
    let base = config_path.parent().context("configuration parent missing")?;
    let config: DriverConfig = serde_json::from_reader(File::open(&config_path)?)?;
    ensure!(config.schema_version == 1, "unsupported schema_version");
    ensure!((1..=2).contains(&config.models.len()), "exactly one or two models required");
    ensure!((1..=256).contains(&config.batch_size), "batch_size must be 1..256");
    ensure!(config.threshold.is_finite(), "threshold must be a finite f64");
    config.preprocessing_options.validate()?;
    if matches!(config.threshold_status, ThresholdStatus::DevelopmentOnly) {
        eprintln!("Using an explicitly DEVELOPMENT-ONLY threshold; this is not a final calibrated submission policy.");
    }
    let query = read_rows(&resolve(base, &config.query_csv))?;
    let gallery = read_rows(&resolve(base, &config.gallery_csv))?;
    ensure!(gallery.len() >= 10, "at least ten gallery rows required");
    let query_ids: Vec<_> = query.iter().map(|row| row.image_id.clone()).collect();
    let gallery_ids: Vec<_> = gallery.iter().map(|row| row.image_id.clone()).collect();
    let mut unique = std::collections::HashSet::new();
    ensure!(query_ids.iter().chain(&gallery_ids).all(|id| unique.insert(id)), "duplicate ID across query/gallery CSVs");
    let runtime = config.runtime.verified_path(base)?;
    let first = &config.models[0];
    first.preprocessing.validate()?;
    let first_graph = first.graph.verified_path(base)?;
    let first_policy = first.metadata_policy.as_ref().map(|p| p.verified_path(base)).transpose()?;
    let mut extractor = match first_policy {
        Some(path) => Extractor::load_with_metadata_policy(&first_graph, &runtime, first.preprocessing.clone(), &path)?,
        None => Extractor::load(&first_graph, &runtime, first.preprocessing.clone())?,
    };
    if config.models.len() == 2 {
        let weight = config.first_weight.context("two models require explicit first_weight")?;
        ensure!(weight.is_finite() && (0.0..=1.0).contains(&weight), "first_weight must be finite in [0,1]");
        let second = &config.models[1];
        second.preprocessing.validate()?;
        let graph = second.graph.verified_path(base)?;
        extractor = match &second.metadata_policy {
            Some(policy) => extractor.with_second_model_and_metadata_policy(&graph, second.preprocessing.clone(), weight, &policy.verified_path(base)?)?,
            None => extractor.with_second_model(&graph, second.preprocessing.clone(), weight)?,
        };
    } else { ensure!(config.first_weight.is_none(), "first_weight applies only to two models"); }
    extractor = extractor.with_preprocessing_options(config.preprocessing_options)?;
    let images = resolve(base, &config.images_dir);
    let query_inputs = inputs(&query, &images);
    let gallery_inputs = inputs(&gallery, &images);
    let mut matrix = Embeddings { rows: 0, dimensions: 0, data: Vec::new() };
    extractor.extract_batches(role_batches(&query_inputs, &gallery_inputs, config.batch_size)?, config.batch_size, |result| {
        if matrix.dimensions == 0 { matrix.dimensions = result.dimensions; }
        if result.dimensions != matrix.dimensions || result.rows.checked_mul(result.dimensions) != Some(result.data.len()) {
            return Err(lct_inference::Error::Invalid("invalid extractor matrix".into()));
        }
        matrix.rows += result.rows;
        matrix.data.extend(result.data);
        Ok(true)
    })?;
    ensure!(matrix.rows == query_inputs.len() + gallery_inputs.len(), "extractor changed row count");
    write_submission(&query_ids, &gallery_ids, &matrix, config.threshold, &output)?;
    println!("Wrote {} queries, {} gallery rows, {}D embeddings to {}", query.len(), gallery.len(), matrix.dimensions, output.display());
    Ok(())
}
