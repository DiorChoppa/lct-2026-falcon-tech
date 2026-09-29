//! reid-cli: пакетный режим без сети. Использует библиотеку inference
//! напрямую, поэтому артефакты считаются тем же кодом, что и gRPC-сервис.
//!
//! Артефакт сдачи формирует Python-пайплайн (`ml/extractor.py` + `reid.submit`,
//! ADR-003); `reid-cli submit` даёт те же три файла с Rust-препроцессингом и
//! служит проверкой паритета сервиса с артефактом (`python -m reid.compare`).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, Subcommand};
use common::{rank, BBox};
use inference::{Crop, ModelManifest, OrtEmbeddingRepository, UseCases};

const TOP_K: usize = 10;

/// Submission policy is separate from the raw-cosine live-service threshold.
fn retrieval_policy(
    json: &serde_json::Value,
    override_threshold: Option<f32>,
) -> anyhow::Result<(bool, f32)> {
    let retrieval = &json["retrieval"];
    let method = retrieval["method"].as_str().unwrap_or("cosine");
    anyhow::ensure!(
        matches!(method, "cosine" | "gallery_dba"),
        "unsupported retrieval method: {method}"
    );
    let use_dba = method == "gallery_dba";
    if use_dba {
        anyhow::ensure!(
            retrieval["k"].as_u64() == Some(4) && retrieval["alpha"].as_f64() == Some(2.0),
            "qualified gallery DBA requires k=4 and alpha=2"
        );
    }
    let threshold = match override_threshold {
        Some(value) => value,
        None => (if use_dba {
            &retrieval["threshold"]
        } else {
            &json["threshold"]
        })
        .as_f64()
        .context("manifest has no retrieval threshold; supply --threshold")? as f32,
    };
    anyhow::ensure!(threshold.is_finite(), "threshold must be finite");
    Ok((use_dba, threshold))
}

/// Refine only the immutable gallery: top five including self, cosine^2 weights.
/// This matches the qualified native helper; original vectors remain unchanged.
fn gallery_dba(gallery: &[Vec<f32>]) -> anyhow::Result<Vec<Vec<f32>>> {
    let dim = gallery.first().context("gallery must not be empty")?.len();
    anyhow::ensure!(dim > 0, "gallery vectors must not be empty");
    for row in gallery {
        let norm = row
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        anyhow::ensure!(
            row.len() == dim && row.iter().all(|v| v.is_finite()) && (norm - 1.0).abs() <= 1e-5,
            "gallery requires finite unit vectors of equal size"
        );
    }
    gallery
        .iter()
        .map(|query| {
            // One small static-gallery sort per row; no quadratic scratch matrix.
            let neighbours = rank(query, gallery, 5);
            let mut pooled = vec![0.0_f64; dim];
            for hit in neighbours {
                let weight = f64::from(hit.score.max(0.0)).powi(2);
                for (acc, &value) in pooled.iter_mut().zip(&gallery[hit.index]) {
                    *acc += weight * f64::from(value);
                }
            }
            let norm = pooled.iter().map(|v| v * v).sum::<f64>().sqrt();
            anyhow::ensure!(
                norm.is_finite() && norm > 0.0,
                "invalid refined gallery vector"
            );
            Ok(pooled.into_iter().map(|v| (v / norm) as f32).collect())
        })
        .collect()
}

#[derive(Parser)]
#[command(name = "reid-cli", about = "Пакетный инференс и артефакты сдачи")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Сформировать submission.csv, embeddings.npy и candidates.csv.
    Submit {
        #[arg(long, default_value = "dataset/images")]
        images: PathBuf,
        #[arg(long, default_value = "dataset/test_query.csv")]
        query: PathBuf,
        #[arg(long, default_value = "dataset/test_gallery.csv")]
        gallery: PathBuf,
        #[arg(long, default_value = "models/model.json")]
        manifest: PathBuf,
        #[arg(long, default_value = "submission-rust")]
        out: PathBuf,
        /// Порог режима отказа по косинусу; по умолчанию `threshold` из манифеста.
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Submit {
            images,
            query,
            gallery,
            manifest,
            out,
            threshold,
            batch_size,
        } => submit(
            &images,
            &query,
            &gallery,
            &manifest,
            &out,
            threshold,
            batch_size.max(1),
        ),
    }
}

struct Row {
    image_id: String,
    bbox: BBox,
}

/// `image_id,x,y,w,h` — как в `test_query.csv` / `test_gallery.csv`.
fn read_rows(path: &Path) -> anyhow::Result<Vec<Row>> {
    let text = fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().context("пустой CSV")?;
    anyhow::ensure!(
        header.trim() == "image_id,x,y,w,h",
        "{}: ожидался заголовок image_id,x,y,w,h, получен {header}",
        path.display()
    );
    lines
        .enumerate()
        .map(|(i, line)| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            anyhow::ensure!(f.len() == 5, "{}: строка {}: {line}", path.display(), i + 2);
            let num = |s: &str| -> anyhow::Result<u32> {
                // В разметке bbox выходит за кадр на 1–2 px; отрицательные — в 0,
                // остальное обрежет inference по размеру кадра.
                Ok(s.parse::<i64>()?.max(0) as u32)
            };
            Ok(Row {
                image_id: f[0].to_string(),
                bbox: BBox {
                    x: num(f[1])?,
                    y: num(f[2])?,
                    w: num(f[3])?,
                    h: num(f[4])?,
                },
            })
        })
        .collect()
}

fn embed_rows(
    use_cases: &UseCases,
    rows: &[Row],
    images: &Path,
    batch_size: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(batch_size) {
        let crops = chunk
            .iter()
            .map(|r| {
                let path = images.join(format!("{}.jpg", r.image_id));
                let bytes = fs::read(&path).with_context(|| path.display().to_string())?;
                Ok(Crop {
                    image: Arc::from(bytes),
                    bbox: r.bbox,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        out.extend(
            use_cases
                .embed(&crops, false)?
                .into_iter()
                .map(|e| e.embedding),
        );
    }
    Ok(out)
}

fn submit(
    images: &Path,
    query: &Path,
    gallery: &Path,
    manifest_path: &Path,
    out: &Path,
    threshold: Option<f32>,
    batch_size: usize,
) -> anyhow::Result<()> {
    let manifest = ModelManifest::load(manifest_path)?;
    let json: serde_json::Value = serde_json::from_str(&fs::read_to_string(manifest_path)?)?;
    let (use_dba, threshold) = retrieval_policy(&json, threshold)?;
    let model_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let repository = Arc::new(OrtEmbeddingRepository::new(&manifest, model_dir));
    let use_cases = UseCases::new(manifest.clone(), repository);
    tracing::info!(model = %manifest.name, version = %manifest.version, threshold, "submit");

    let q_rows = read_rows(query)?;
    let g_rows = read_rows(gallery)?;
    anyhow::ensure!(!g_rows.is_empty(), "gallery must not be empty");
    let t0 = Instant::now();
    let q_emb = embed_rows(&use_cases, &q_rows, images, batch_size)?;
    let g_emb = embed_rows(&use_cases, &g_rows, images, batch_size)?;
    let secs = t0.elapsed().as_secs_f64();
    let n = q_rows.len() + g_rows.len();
    tracing::info!(
        n,
        secs = format!("{secs:.1}"),
        per_sec = format!("{:.1}", n as f64 / secs),
        "кропы"
    );

    fs::create_dir_all(out)?;
    write_npy(
        &out.join("embeddings.npy"),
        &q_emb,
        &g_emb,
        manifest.dim as usize,
    )?;

    let refined = if use_dba {
        Some(gallery_dba(&g_emb)?)
    } else {
        None
    };
    let search_gallery = refined.as_deref().unwrap_or(&g_emb);

    let mut submission = fs::File::create(out.join("submission.csv"))?;
    let mut candidates = fs::File::create(out.join("candidates.csv"))?;
    writeln!(candidates, "query_id,gallery_id,confidence")?;
    for (row, emb) in q_rows.iter().zip(&q_emb) {
        let ranked = rank(emb, search_gallery, TOP_K);
        let ids: Vec<&str> = ranked
            .iter()
            .map(|r| g_rows[r.index].image_id.as_str())
            .collect();
        writeln!(submission, "{},{}", row.image_id, ids.join(","))?;
        if let Some(top) = ranked.first() {
            if top.score >= threshold {
                writeln!(
                    candidates,
                    "{},{},{:.6}",
                    row.image_id, g_rows[top.index].image_id, top.score
                )?;
            }
        }
    }
    fs::write(
        out.join("run.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "model": manifest.name,
            "version": manifest.version,
            "provider": "rust-ort",
            "threshold": threshold,
            "retrieval": if use_dba { "gallery_dba_k4_alpha2" } else { "cosine" },
            "n_query": q_rows.len(),
            "n_gallery": g_rows.len(),
            "batch_size": batch_size,
            "seconds": (secs * 100.0).round() / 100.0,
        }))?,
    )?;
    tracing::info!(out = %out.display(), "готово");
    Ok(())
}

/// `.npy` версии 1.0: float32 little-endian, shape (n, dim), сначала query, затем gallery.
fn write_npy(path: &Path, q: &[Vec<f32>], g: &[Vec<f32>], dim: usize) -> anyhow::Result<()> {
    let n = q.len() + g.len();
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {dim}), }}");
    // Заголовок вместе с магией (10 байт) выравнивается на 64 и заканчивается '\n'.
    let pad = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(pad % 64));
    header.push('\n');
    let mut f = fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for row in q.iter().chain(g) {
        anyhow::ensure!(
            row.len() == dim,
            "эмбеддинг длины {}, ожидалось {dim}",
            row.len()
        );
        for v in row {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dba_uses_static_gallery_and_preserves_original_vectors() {
        let gallery = vec![vec![1.0, 0.0], vec![0.8, 0.6], vec![0.0, 1.0]];
        let before = gallery.clone();
        let result = gallery_dba(&gallery).unwrap();
        assert!((result[0][0] - 0.9692308).abs() < 1e-6);
        assert!((result[0][1] - 0.2461538).abs() < 1e-6);
        assert_eq!(gallery, before);
        let first = rank(&[1.0, 0.0], &result, 10);
        let _ = rank(&[0.0, 1.0], &result, 10);
        assert_eq!(first, rank(&[1.0, 0.0], &result, 10));
        assert!(gallery_dba(&[]).is_err());
    }

    #[test]
    fn submission_threshold_and_explicit_override_are_separate_from_service() {
        let json = serde_json::json!({"threshold": 0.4, "retrieval": {
            "method": "gallery_dba", "k": 4, "alpha": 2, "threshold": 0.7
        }});
        assert_eq!(retrieval_policy(&json, None).unwrap(), (true, 0.7));
        assert_eq!(retrieval_policy(&json, Some(0.8)).unwrap(), (true, 0.8));
        assert!(retrieval_policy(&json, Some(f32::NAN)).is_err());
    }

    #[test]
    fn ranking_exact_ties_keep_gallery_order() {
        let gallery = gallery_dba(&vec![vec![1.0, 0.0]; 12]).unwrap();
        let indices: Vec<_> = rank(&[1.0, 0.0], &gallery, 10)
            .iter()
            .map(|h| h.index)
            .collect();
        assert_eq!(indices, (0..10).collect::<Vec<_>>());
    }
}
