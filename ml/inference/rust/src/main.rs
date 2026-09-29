use anyhow::{Context, Result, ensure};
use lct_inference::{BBox, Config, Extractor, Input, PreprocessingOptions, preprocess};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Deserialize)]
struct Row {
    image_id: String,
    path: PathBuf,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

struct Fusion {
    model: PathBuf,
    config_path: PathBuf,
    config: Config,
    first_weight: f32,
}

#[derive(Default)]
struct RuntimePolicies {
    first: Option<PathBuf>,
    second: Option<PathBuf>,
}

impl RuntimePolicies {
    fn any(&self) -> bool {
        self.first.is_some() || self.second.is_some()
    }
}

fn parse_policies(options: &[String]) -> Result<(Vec<String>, RuntimePolicies)> {
    ensure!(options.len().is_multiple_of(2), "flags require values");
    let mut policies = RuntimePolicies::default();
    let mut remaining = Vec::new();
    for pair in options.chunks_exact(2) {
        let destination = match pair[0].as_str() {
            "--metadata-policy" => Some(&mut policies.first),
            "--second-metadata-policy" => Some(&mut policies.second),
            _ => None,
        };
        if let Some(destination) = destination {
            ensure!(destination.is_none(), "duplicate metadata policy flag");
            *destination = Some(PathBuf::from(&pair[1]));
        } else {
            remaining.extend_from_slice(pair);
        }
    }
    Ok((remaining, policies))
}

fn parse_preprocessing(options: &[String]) -> Result<(Vec<String>, PreprocessingOptions)> {
    ensure!(options.len().is_multiple_of(2), "flags require values");
    let (mut workers, mut reuse) = (None, None);
    let mut remaining = Vec::new();
    for pair in options.chunks_exact(2) {
        match pair[0].as_str() {
            "--preprocess-workers" => {
                ensure!(workers.is_none(), "duplicate --preprocess-workers");
                workers = Some(pair[1].parse::<usize>()?);
            }
            "--reuse-identical-preprocessing" => {
                ensure!(reuse.is_none(), "duplicate --reuse-identical-preprocessing");
                reuse = Some(pair[1].parse::<bool>()?);
            }
            _ => remaining.extend_from_slice(pair),
        }
    }
    let options = PreprocessingOptions {
        workers: workers.unwrap_or(1),
        reuse_identical: reuse.unwrap_or(false),
    };
    options.validate()?;
    Ok((remaining, options))
}

fn parse_fusion(options: &[String]) -> Result<Option<Fusion>> {
    if options.is_empty() {
        return Ok(None);
    }
    ensure!(
        options.len().is_multiple_of(2),
        "ensemble flags require values"
    );
    let (mut model, mut config_path, mut weight) = (None, None, None);
    for pair in options.chunks_exact(2) {
        match pair[0].as_str() {
            "--second-model" => {
                ensure!(model.is_none(), "duplicate --second-model");
                model = Some(PathBuf::from(&pair[1]));
            }
            "--second-config" => {
                ensure!(config_path.is_none(), "duplicate --second-config");
                config_path = Some(PathBuf::from(&pair[1]));
            }
            "--fusion-weight" => {
                ensure!(weight.is_none(), "duplicate --fusion-weight");
                weight = Some(pair[1].parse::<f32>()?);
            }
            flag => anyhow::bail!("unknown ensemble flag: {flag}"),
        }
    }
    let model = model.context("ensemble requires --second-model")?;
    let config_path = config_path.context("ensemble requires --second-config")?;
    let first_weight = weight.unwrap_or(0.5);
    ensure!(
        first_weight.is_finite() && (0.0..=1.0).contains(&first_weight),
        "fusion weight must be finite in [0,1]"
    );
    let config: Config = serde_json::from_reader(File::open(&config_path)?)?;
    config.validate()?;
    Ok(Some(Fusion {
        model,
        config_path,
        config,
        first_weight,
    }))
}

fn load_extractor(
    model: &Path,
    runtime: &Path,
    config: Config,
    fusion: Option<&Fusion>,
    policies: &RuntimePolicies,
    preprocessing: PreprocessingOptions,
) -> Result<Extractor> {
    let extractor = if let Some(policy) = &policies.first {
        Extractor::load_with_metadata_policy(model, runtime, config, policy)?
    } else {
        Extractor::load(model, runtime, config)?
    };
    let extractor = if let Some(fusion) = fusion {
        if let Some(policy) = &policies.second {
            extractor.with_second_model_and_metadata_policy(
                &fusion.model,
                fusion.config.clone(),
                fusion.first_weight,
                policy,
            )?
        } else {
            extractor.with_second_model(
                &fusion.model,
                fusion.config.clone(),
                fusion.first_weight,
            )?
        }
    } else {
        extractor
    };
    Ok(extractor.with_preprocessing_options(preprocessing)?)
}

fn fusion_metadata(fusion: Option<&Fusion>) -> serde_json::Value {
    match fusion {
        None => serde_json::Value::Null,
        Some(fusion) => serde_json::json!({
            "second_model": fusion.model, "second_config_file": fusion.config_path,
            "second_config": fusion.config, "first_weight": fusion.first_weight,
            "second_weight": 1.0 - fusion.first_weight,
            "method": "L2(concat(sqrt(w)*L2(first),sqrt(1-w)*L2(second)))",
            "shared_decode_crop": true, "execution": "sequential CUDA device0",
        }),
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let flag_start = args
        .iter()
        .position(|arg| arg.starts_with("--"))
        .unwrap_or(args.len());
    let (args, options) = args.split_at(flag_start);
    ensure!(
        args.len() >= 4,
        "usage: lct-inference preprocess CONFIG MANIFEST OUTPUT_PREFIX | extract CONFIG MODEL MANIFEST OUTPUT_PREFIX [BATCH_SIZE] | benchmark CONFIG MODEL MANIFEST OUTPUT_JSON; optionally append --second-model MODEL --second-config CONFIG [--fusion-weight FIRST_WEIGHT], --metadata-policy POLICY and/or --second-metadata-policy POLICY, --preprocess-workers 1..4, --reuse-identical-preprocessing true|false"
    );
    let extract = args[0] == "extract";
    let benchmark_mode = args[0] == "benchmark";
    ensure!(
        extract || benchmark_mode || options.is_empty(),
        "ensemble flags apply only to extract/benchmark"
    );
    ensure!(
        extract || benchmark_mode || args[0] == "preprocess",
        "unknown command"
    );
    ensure!(
        (args[0] == "preprocess" && args.len() == 4)
            || (extract && (args.len() == 5 || args.len() == 6))
            || (benchmark_mode && args.len() == 5),
        "invalid argument count"
    );
    let config: Config = serde_json::from_reader(File::open(&args[1])?)?;
    config.validate()?;
    let (fusion_options, policies) = parse_policies(options)?;
    let (fusion_options, preprocessing) = parse_preprocessing(&fusion_options)?;
    let fusion = parse_fusion(&fusion_options)?;
    ensure!(
        policies.second.is_none() || fusion.is_some(),
        "second metadata policy requires second model/config"
    );
    ensure!(
        !policies.any() || env::var_os("ORT_PROFILE_PREFIX").is_none(),
        "unset ORT_PROFILE_PREFIX for metadata policy; use ORT_POLICY_PROFILE_DIR for startup evidence"
    );
    let manifest_index = if extract || benchmark_mode { 3 } else { 2 };
    let rows: Vec<Row> = csv::Reader::from_path(&args[manifest_index])?
        .deserialize()
        .collect::<std::result::Result<_, _>>()?;
    ensure!(!rows.is_empty(), "empty manifest");
    let inputs: Vec<_> = rows
        .iter()
        .map(|r| Input {
            path: r.path.clone(),
            bbox: BBox {
                x: r.x,
                y: r.y,
                w: r.w,
                h: r.h,
            },
        })
        .collect();
    if benchmark_mode {
        return benchmark(
            args,
            config,
            &rows,
            &inputs,
            fusion.as_ref(),
            &policies,
            preprocessing,
        );
    }
    let prefix = &args[manifest_index + 1];
    let raw_path = format!("{prefix}.f32");
    let sidecar_path = format!("{prefix}.json");
    ensure!(
        !Path::new(&raw_path).exists() && !Path::new(&sidecar_path).exists(),
        "output already exists; choose a new prefix"
    );
    let mut output = BufWriter::new(File::create(&raw_path)?);
    let mut extraction = if extract {
        Some(load_extractor(
            Path::new(&args[2]),
            Path::new(&env::var("ORT_DYLIB_PATH").context("set ORT_DYLIB_PATH")?),
            config.clone(),
            fusion.as_ref(),
            &policies,
            preprocessing,
        )?)
    } else {
        None
    };
    let batch_size = if args.len() == 6 {
        args[5].parse::<usize>()?
    } else {
        1
    };
    ensure!((1..=256).contains(&batch_size), "batch size must be 1..256");
    let start = Instant::now();
    let mut width = 0;
    for chunk in inputs.chunks(batch_size) {
        let data = if let Some(extractor) = &mut extraction {
            let embeddings = extractor.extract_batch(chunk)?;
            if width != 0 {
                ensure!(width == embeddings.dimensions, "embedding width changed");
            }
            width = embeddings.dimensions;
            embeddings.data
        } else {
            let mut data = Vec::new();
            for input in chunk {
                data.extend(preprocess(input, &config)?);
            }
            data
        };
        for value in data {
            output.write_all(&value.to_le_bytes())?;
        }
    }
    output.flush()?;
    if env::var_os("ORT_PROFILE_PREFIX").is_some()
        && let Some(extractor) = &mut extraction
    {
        eprintln!("ORT profile: {}", extractor.end_profiling()?);
    }
    let shape = if extract {
        vec![rows.len(), width]
    } else {
        vec![rows.len(), 3, config.size as usize, config.size as usize]
    };
    let sidecar = serde_json::json!({"shape":shape,"dtype":"float32","byte_order":"little","layout":if extract{"ND"}else{"NCHW"},"image_ids":rows.iter().map(|r|&r.image_id).collect::<Vec<_>>(),"config":config,"fusion":fusion_metadata(fusion.as_ref()),"preprocessing_options":preprocessing,"runtime_policy":extraction.as_ref().map(Extractor::runtime_policy_evidence),"seconds":start.elapsed().as_secs_f64(),"includes_output_serialization":true,"mode":args[0]});
    serde_json::to_writer_pretty(File::create(sidecar_path)?, &sidecar)?;
    eprintln!("Wrote {} rows to {raw_path}", rows.len());
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("hashing {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let index = quantile * (ordered.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    ordered[lower] + (ordered[upper] - ordered[lower]) * index.fract()
}

fn next_batch(inputs: &[Input], cursor: &mut usize, size: usize) -> Vec<Input> {
    let batch = (0..size)
        .map(|offset| inputs[(*cursor + offset) % inputs.len()].clone())
        .collect();
    *cursor = (*cursor + size) % inputs.len();
    batch
}

fn benchmark(
    args: &[String],
    config: Config,
    rows: &[Row],
    inputs: &[Input],
    fusion: Option<&Fusion>,
    policies: &RuntimePolicies,
    preprocessing: PreprocessingOptions,
) -> Result<()> {
    let started_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    let output_path = Path::new(&args[4]);
    ensure!(!output_path.exists(), "benchmark output already exists");
    ensure!(
        inputs.len() >= 32,
        "benchmark requires at least32 manifest rows"
    );
    ensure!(
        inputs
            .iter()
            .all(|input| input.path.extension().is_some_and(|extension| {
                extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
            })),
        "benchmark requires original JPEG paths, not decoded crops/tensors"
    );
    ensure!(
        env::var_os("ORT_PROFILE_PREFIX").is_none(),
        "unset ORT_PROFILE_PREFIX for scoring timings; profile in a separate extract run"
    );
    let runtime = PathBuf::from(env::var_os("ORT_DYLIB_PATH").context("set ORT_DYLIB_PATH")?);
    let model = Path::new(&args[2]);
    // Provenance, model loading and JSON serialization are all outside measured regions.
    let sources = serde_json::json!({
        "src/lib.rs": format!("{:x}", Sha256::digest(include_bytes!("lib.rs"))),
        "src/main.rs": format!("{:x}", Sha256::digest(include_bytes!("main.rs"))),
        "src/resize.rs": format!("{:x}", Sha256::digest(include_bytes!("resize.rs"))),
        "src/jpeg.rs": format!("{:x}", Sha256::digest(include_bytes!("jpeg.rs"))),
        "src/metadata.rs": format!("{:x}", Sha256::digest(include_bytes!("metadata.rs"))),
        "src/prefetch.rs": format!("{:x}", Sha256::digest(include_bytes!("prefetch.rs"))),
        "src/batch.rs": format!("{:x}", Sha256::digest(include_bytes!("batch.rs"))),
        "Cargo.toml": format!("{:x}", Sha256::digest(include_bytes!("../Cargo.toml"))),
        "Cargo.lock": format!("{:x}", Sha256::digest(include_bytes!("../Cargo.lock"))),
    });
    let hashes = serde_json::json!({
        "model": file_sha256(model)?, "native_runtime": file_sha256(&runtime)?,
        "manifest": file_sha256(Path::new(&args[3]))?,
        "config_file": file_sha256(Path::new(&args[1]))?,
        "executable": file_sha256(&env::current_exe()?)?, "source": sources,
        "second_model": fusion.map(|f| file_sha256(&f.model)).transpose()?,
        "second_config_file": fusion.map(|f| file_sha256(&f.config_path)).transpose()?,
        "metadata_policy": policies.first.as_ref().map(|path| file_sha256(path)).transpose()?,
        "second_metadata_policy": policies.second.as_ref().map(|path| file_sha256(path)).transpose()?,
    });
    let load_start = Instant::now();
    let mut extractor = load_extractor(
        model,
        &runtime,
        config.clone(),
        fusion,
        policies,
        preprocessing,
    )?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let mut cursor = 0;
    let mut dimensions = 0;
    let mut latency_warmup_ms = Vec::with_capacity(50);
    for _ in 0..50 {
        let start = Instant::now();
        let output = extractor.extract_batch(&next_batch(inputs, &mut cursor, 1))?;
        latency_warmup_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        dimensions = output.dimensions;
        std::hint::black_box(output);
    }
    let mut latency_ms = Vec::with_capacity(300);
    for _ in 0..300 {
        let batch = next_batch(inputs, &mut cursor, 1);
        let start = Instant::now();
        let output = extractor.extract_batch(&batch)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        ensure!(output.dimensions == dimensions, "embedding width changed");
        std::hint::black_box(output);
        latency_ms.push(elapsed_ms);
    }
    let mut throughput = Vec::new();
    let mut failed_batch_sizes = Vec::new();
    for batch_size in [1, 8, 16, 32] {
        let warmup_start = Instant::now();
        let mut warmup_ms = Vec::with_capacity(50);
        let mut warmup_failure = None;
        for _ in 0..50 {
            let batch = next_batch(inputs, &mut cursor, batch_size);
            let start = Instant::now();
            let output = match extractor.extract_batch(&batch) {
                Ok(output) => output,
                Err(error) => {
                    warmup_failure = Some(error.to_string());
                    break;
                }
            };
            warmup_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            ensure!(output.dimensions == dimensions, "embedding width changed");
            std::hint::black_box(output);
        }
        let warmup_seconds = warmup_start.elapsed().as_secs_f64();
        if let Some(error) = warmup_failure {
            failed_batch_sizes.push(batch_size);
            throughput.push(serde_json::json!({"status":"failed","phase":"warmup","batch_size":batch_size,"error":error,"warmup_wall_seconds":warmup_seconds,"warmup_samples_ms":warmup_ms}));
            continue;
        }
        let mut batch_ms = Vec::new();
        let mut failure = None;
        let start = Instant::now();
        let mut previous_completion = start;
        let batches = std::iter::from_fn(|| Some(next_batch(inputs, &mut cursor, batch_size)));
        let extraction = extractor.extract_batches(batches, batch_size, |output| {
            if output.dimensions != dimensions {
                return Err(lct_inference::Error::Invalid("embedding dimensions changed".into()));
            }
            let completed = Instant::now();
            batch_ms.push((completed - previous_completion).as_secs_f64() * 1000.0);
            previous_completion = completed;
            std::hint::black_box(output);
            Ok(start.elapsed().as_secs_f64() < 10.0 || batch_ms.len() < 3)
        });
        if let Err(error) = extraction { failure = Some(error.to_string()); }
        // extract_batches returns only after receiver cancellation and producer join.
        let seconds = start.elapsed().as_secs_f64();
        if let Some(error) = failure {
            failed_batch_sizes.push(batch_size);
            throughput.push(serde_json::json!({
                "status": "failed", "batch_size": batch_size, "error": error,
                "phase":"measurement", "warmup_wall_seconds":warmup_seconds,"warmup_samples_ms":warmup_ms,
                "completed_batches": batch_ms.len(), "wall_seconds": seconds,
                "batch_samples_ms": batch_ms,
            }));
            continue;
        }
        throughput.push(serde_json::json!({
            "status": "completed",
            "batch_size": batch_size, "batches": batch_ms.len(),
            "images": batch_ms.len() * batch_size, "wall_seconds": seconds,
            "fps": (batch_ms.len() * batch_size) as f64 / seconds,
            "warmup_wall_seconds":warmup_seconds,"warmup_samples_ms":warmup_ms,
            "first_three_shape_calls_ms":&warmup_ms[..3],
            "batch_median_ms": percentile(&batch_ms, 0.5),
            "batch_p95_ms": percentile(&batch_ms, 0.95), "batch_samples_ms": batch_ms,
        }));
    }
    let best_fps = throughput
        .iter()
        .filter_map(|run| run["fps"].as_f64())
        .fold(0.0, f64::max);
    let result = serde_json::json!({
        "status": if failed_batch_sizes.is_empty() { "completed" } else { "completed_with_failed_batch_sizes" },
        "kind": "native_rust_full_extraction_benchmark", "started_unix_seconds": started_unix_seconds,
        "failed_batch_sizes": failed_batch_sizes,
        "protocol": {"source": "jury QA Q30/Q31/Q34", "warmups_batch1": 50,
            "throughput_pipeline":"one future CPU batch; wall includes fill, pending work and producer join; B1 synchronous", "throughput_batch_samples":"ordered output inter-completion intervals, not individual request latency",
        "throughput_warmups_per_shape":50,"throughput_minimum_completed_batches":3,
            "latency_runs_batch1": 300, "throughput_minimum_seconds_per_batch": 10,
            "boundary": "read/decode/bbox/resize/normalize/forward/host-output/L2; optional second transform/forward and weighted-concat/L2 included",
            "excluded": "model startup, provenance hashing, JSON serialization, search/reranking",
            "synchronization": "default synchronous ORT Session::run; CPU output; no async/IObinding or disable_synchronize_execution_providers",
            "throughput_wall_includes": "batch bookkeeping, input metadata cloning and output disposal",
            "image_sequence": "manifest order, cyclic; ordered CPU preparation; no decoded image cache"},
        "runtime": {"ort_crate": "2.0.0-rc.12", "native_expected": "1.24.4/API24",
            "native_library": runtime, "provider": if policies.any() { "CUDA+audited_CPU_metadata" } else { "CUDA" }, "device_id": 0,
            "cpu_fallback": if policies.any() { serde_json::Value::Null } else { serde_json::json!(false) },
            "cpu_execution_policy": if policies.any() { "explicit_graph_bound_metadata_exceptions" } else { "disabled" },
            "tf32": true, "input_dtype": "float32",
            "output_dtype": "float32", "internal_graph_precision": "not inferred; graph identified by model SHA256"},
        "config": config, "model": model, "manifest": args[3], "fusion": fusion_metadata(fusion),
        "preprocessing_options":preprocessing,"latency_warmup_samples_ms":latency_warmup_ms,
        "runtime_policy": extractor.runtime_policy_evidence(),
        "image_ids": rows.iter().map(|row| &row.image_id).collect::<Vec<_>>(),
        "input_shape": ["N", 3, config.size, config.size],
        "embedding_dimensions": dimensions, "model_load_seconds": load_seconds,
        "latency_median_ms": percentile(&latency_ms, 0.5),
        "latency_p95_ms": percentile(&latency_ms, 0.95), "latency_samples_ms": latency_ms,
        "throughput": throughput, "best_fps": best_fps, "sha256": hashes,
        "peak_vram_bytes": null, "peak_vram_status": "not measured by this CLI",
    });
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &result)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocessing_flags_preserve_defaults_and_reject_invalid_or_duplicate_values() {
        assert_eq!(
            parse_preprocessing(&[]).unwrap().1,
            PreprocessingOptions::default()
        );
        let options = [
            "--preprocess-workers",
            "4",
            "--reuse-identical-preprocessing",
            "true",
        ]
        .map(String::from);
        assert_eq!(
            parse_preprocessing(&options).unwrap().1,
            PreprocessingOptions {
                workers: 4,
                reuse_identical: true
            }
        );
        for values in [
            vec!["--preprocess-workers", "0"],
            vec!["--preprocess-workers", "5"],
            vec!["--reuse-identical-preprocessing", "yes"],
            vec!["--preprocess-workers", "1", "--preprocess-workers", "4"],
        ] {
            assert!(
                parse_preprocessing(&values.into_iter().map(String::from).collect::<Vec<_>>())
                    .is_err()
            );
        }
    }

    #[test]
    fn policy_flags_are_explicit_independent_and_reject_duplicates() {
        let options: Vec<_> = [
            "--metadata-policy",
            "first.json",
            "--second-model",
            "second.onnx",
            "--second-metadata-policy",
            "second.json",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let (remaining, policies) = parse_policies(&options).unwrap();
        assert_eq!(policies.first, Some(PathBuf::from("first.json")));
        assert_eq!(policies.second, Some(PathBuf::from("second.json")));
        assert_eq!(remaining, vec!["--second-model", "second.onnx"]);
        let duplicate: Vec<_> = ["--metadata-policy", "a", "--metadata-policy", "b"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert!(parse_policies(&duplicate).is_err());
        assert!(!parse_policies(&[]).unwrap().1.any());
    }

    #[test]
    fn ensemble_flags_reject_incomplete_duplicate_and_invalid_values_before_runtime_load() {
        for options in [
            vec!["--second-model"],
            vec!["--fusion-weight", "0.5"],
            vec!["--second-model", "a", "--second-model", "b"],
            vec!["--unknown", "x"],
            vec![
                "--second-model",
                "a",
                "--second-config",
                "b",
                "--fusion-weight",
                "NaN",
            ],
            vec![
                "--second-model",
                "a",
                "--second-config",
                "b",
                "--fusion-weight",
                "-0.1",
            ],
        ] {
            let args: Vec<_> = options.into_iter().map(str::to_string).collect();
            assert!(parse_fusion(&args).is_err());
        }
        assert!(parse_fusion(&[]).unwrap().is_none());
    }

    #[test]
    fn percentiles_match_linear_numpy_default_including_even_median() {
        assert_eq!(percentile(&[4.0, 1.0, 2.0, 3.0], 0.5), 2.5);
        assert!((percentile(&[4.0, 1.0, 2.0, 3.0], 0.95) - 3.85).abs() < 1e-12);
    }
}
