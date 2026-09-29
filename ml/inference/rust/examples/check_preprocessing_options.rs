//! CPU-only fixture parity and phase timings; never initializes ONNX Runtime.
use anyhow::{Result, ensure};
use lct_inference::{BBox, Config, Input, PreprocessingOptions, preprocess_batch};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
struct Row {
    path: PathBuf,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

fn inputs(path: &Path) -> Result<Vec<Input>> {
    csv::Reader::from_path(path)?
        .deserialize::<Row>()
        .map(|row| {
            let row = row?;
            Ok(Input {
                path: row.path,
                bbox: BBox {
                    x: row.x,
                    y: row.y,
                    w: row.w,
                    h: row.h,
                },
            })
        })
        .collect()
}

fn hash(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

fn differences(values: &[f32], expected: &[u8]) -> usize {
    values
        .iter()
        .zip(expected.chunks_exact(4))
        .filter(|(value, bytes)| value.to_bits().to_le_bytes() != **bytes)
        .count()
}

fn main() -> Result<()> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("expected new output JSON path"))?,
    );
    ensure!(!output.exists(), "refuse overwrite");
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/inference_preprocessing");
    let options = [
        PreprocessingOptions::default(),
        PreprocessingOptions {
            workers: 1,
            reuse_identical: true,
        },
        PreprocessingOptions {
            workers: 4,
            reuse_identical: false,
        },
        PreprocessingOptions {
            workers: 4,
            reuse_identical: true,
        },
    ];
    let previous: serde_json::Value = serde_json::from_reader(File::open(
        fixture.join("jpeg_decoder_v2/final/parity.json"),
    )?)?;
    let mut cases = Vec::new();
    for source in ["jpeg", "pillow_png"] {
        let manifest = fixture.join(format!("manifest_{source}.csv"));
        let images = inputs(&manifest)?;
        ensure!(images.len() == 128, "expected exactly128 training fixtures");
        for mode in ["center", "stretch", "letterbox"] {
            let config: Config =
                serde_json::from_reader(File::open(fixture.join(format!("{mode}.json")))?)?;
            let reference_path = fixture.join(format!("jpeg_decoder_v2/final/{mode}_{source}.f32"));
            let reference = fs::read(&reference_path)?;
            let reference_sha = hash(&reference_path)?;
            let manifest_sha = hash(&manifest)?;
            let original = previous["cases"]
                .as_array()
                .and_then(|rows| {
                    rows.iter()
                        .find(|row| row["mode"] == mode && row["source"] == source)
                })
                .ok_or_else(|| anyhow::anyhow!("missing frozen reference record"))?;
            ensure!(
                original["reference_sha256"] == reference_sha
                    && original["manifest_sha256"] == manifest_sha,
                "fixture provenance changed"
            );
            let width = 3 * (config.size as usize).pow(2);
            ensure!(
                reference.len() == 128 * width * 4,
                "invalid reference shape"
            );
            for options in options {
                let (mut first_differences, mut second_differences) = (0, 0);
                let start = Instant::now();
                for (chunk_index, chunk) in images.chunks(16).enumerate() {
                    let tensors = preprocess_batch(chunk, &config, Some(&config), options)?;
                    let offset = chunk_index * 16 * width * 4;
                    let expected = &reference[offset..offset + chunk.len() * width * 4];
                    ensure!(
                        tensors.first.len() * 4 == expected.len(),
                        "first shape changed"
                    );
                    let second = tensors
                        .second
                        .ok_or_else(|| anyhow::anyhow!("second tensor missing"))?;
                    ensure!(second.len() * 4 == expected.len(), "second shape changed");
                    first_differences += differences(&tensors.first, expected);
                    second_differences += differences(&second, expected);
                }
                let record = json!({"source":source,"mode":mode,"options":options,"images":128,
                    "compared_values":128*width*2,"first_changed_values":first_differences,"second_changed_values":second_differences,
                    "seconds_including_bitwise_comparison":start.elapsed().as_secs_f64(),"reference_sha256":reference_sha,"manifest_sha256":manifest_sha});
                eprintln!(
                    "{source}/{mode} workers{} reuse{}: differences {first_differences}/{second_differences}",
                    options.workers, options.reuse_identical
                );
                cases.push(record);
                ensure!(
                    first_differences == 0 && second_differences == 0,
                    "bitwise parity failed"
                );
            }
        }
    }
    let images = inputs(&fixture.join("manifest_jpeg.csv"))?;
    let mut config: Config = serde_json::from_reader(File::open(fixture.join("stretch.json"))?)?;
    config.size = 256;
    let mut timings = Vec::new();
    for size in [1, 32] {
        let mut samples = vec![Vec::new(); options.len()];
        for _ in 0..3 {
            for (index, options) in options.iter().enumerate() {
                let start = Instant::now();
                let tensors = preprocess_batch(&images[..size], &config, Some(&config), *options)?;
                samples[index].push(start.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(tensors);
            }
        }
        for (options, samples) in options.iter().zip(samples) {
            let mut sorted = samples.clone();
            sorted.sort_by(f64::total_cmp);
            timings.push(json!({"batch_size":size,"options":options,"samples_ms":samples,"median_ms":sorted[1],"preprocessing_only_fps":size as f64*1000.0/sorted[1]}));
        }
    }
    let report = json!({"status":"cpu_passed","gpu_executed":false,"cases":cases,"phase_timings":timings,
        "timing_note":"same first32 training fixture JPEGs; CPU decode/crop/transforms/batch assembly only, three interleaved repetitions, not full extraction or jury throughput",
        "timing_config":config,"source_sha256":{"batch.rs":hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/batch.rs"))?,
            "lib.rs":hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))?,"example":hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join(file!()))?}});
    serde_json::to_writer_pretty(
        File::options().write(true).create_new(true).open(output)?,
        &report,
    )?;
    Ok(())
}
