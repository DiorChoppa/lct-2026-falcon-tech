//! Offline CSV input and jury-file output; no network or backend service.
use anyhow::{Context, Result, ensure};
use lct_inference::{BBox, Embeddings, Input};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs::{self, File}, io::{BufWriter, Read, Write}, path::{Path, PathBuf}};

#[path = "../../retrieval.rs"]
pub mod retrieval;

/// Image metadata only. Extra identity/camera columns are deliberately ignored.
#[derive(Debug, Deserialize)]
pub struct Row {
    pub image_id: String,
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
}

fn validate_id(id: &str) -> Result<()> {
    ensure!(!id.is_empty() && id.trim() == id && !matches!(id, "." | ".."), "invalid empty/whitespace image ID");
    ensure!(!id.chars().any(|c| c.is_control() || "<>:\"/\\|?*".contains(c)) && !id.ends_with('.'), "image ID must be one safe filename component");
    Ok(())
}

/// Read canonical CSV rows in their original order; do not use labels.
pub fn read_rows(path: &Path) -> Result<Vec<Row>> {
    let mut reader = csv::ReaderBuilder::new().flexible(false).from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut unique_headers = HashSet::new();
    ensure!(headers.iter().all(|name| unique_headers.insert(name)), "duplicate CSV header");
    for name in ["image_id", "x", "y", "w", "h"] {
        ensure!(headers.iter().any(|header| header == name), "missing CSV column {name}");
    }
    let rows: Vec<Row> = reader.deserialize().collect::<std::result::Result<_, _>>()?;
    ensure!(!rows.is_empty(), "empty image CSV");
    let mut ids = HashSet::new();
    for row in &rows {
        validate_id(&row.image_id)?;
        ensure!(ids.insert(row.image_id.as_str()), "duplicate image ID {}", row.image_id);
        ensure!(row.x >= 0 && row.y >= 0 && row.w > 0 && row.h > 0 && row.x.checked_add(row.w).is_some() && row.y.checked_add(row.h).is_some(), "invalid bbox for {}", row.image_id);
    }
    Ok(rows)
}

/// Preserve row order while resolving the supplied image directory.
pub fn inputs(rows: &[Row], directory: &Path) -> Vec<Input> {
    rows.iter().map(|row| Input { path: directory.join(format!("{}.jpg", row.image_id)),
        bbox: BBox { x: row.x, y: row.y, w: row.w, h: row.h } }).collect()
}

/// Batches never cross the query/gallery boundary, independent of query count.
pub fn role_batches<'a>(query: &'a [Input], gallery: &'a [Input], batch_size: usize)
    -> Result<impl Iterator<Item = &'a [Input]>> {
    ensure!(batch_size > 0, "batch_size must be positive");
    Ok(query.chunks(batch_size).chain(gallery.chunks(batch_size)))
}

/// Keep gallery batch boundaries independent of the number of supplied queries.
/// The returned matrix still contains all query rows followed by all gallery rows.
pub fn extract_separately(
    query: &[Input],
    gallery: &[Input],
    batch_size: usize,
    mut extract_batch: impl FnMut(&[Input]) -> Result<Embeddings>,
) -> Result<Embeddings> {
    ensure!(batch_size > 0, "batch_size must be positive");
    let mut matrix = Embeddings { rows: 0, dimensions: 0, data: Vec::new() };
    for chunk in role_batches(query, gallery, batch_size)? {
            let result = extract_batch(chunk)?;
            ensure!(result.rows == chunk.len(), "extractor changed batch row count");
            if matrix.dimensions == 0 { matrix.dimensions = result.dimensions; }
            ensure!(result.dimensions > 0 && matrix.dimensions == result.dimensions
                && result.rows.checked_mul(result.dimensions) == Some(result.data.len()),
                "extractor changed dimensions or shape");
            matrix.rows += result.rows;
            matrix.data.extend(result.data);
    }
    Ok(matrix)
}

/// Stream a file hash without loading model weights a second time into RAM.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 { break; }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn write_npy(path: &Path, matrix: &Embeddings) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", matrix.rows, matrix.dimensions);
    let padding = (16 - (10 + header.len() + 1) % 16) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let length = u16::try_from(header.len()).context("NPY v1 header too long")?;
    writer.write_all(b"\x93NUMPY\x01\x00")?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    for value in &matrix.data { writer.write_all(&value.to_le_bytes())?; }
    writer.flush()?;
    Ok(())
}

/// Pure CPU search/output over already extracted vectors. No renormalization.
/// The directory must not exist. Invalid vectors are rejected before creation.
pub fn write_submission(query_ids: &[String], gallery_ids: &[String], matrix: &Embeddings, threshold: f64, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "refusing to overwrite output directory {}", output.display());
    ensure!(!query_ids.is_empty() && gallery_ids.len() >= 10, "need queries and at least ten gallery images");
    let total_rows = query_ids.len().checked_add(gallery_ids.len()).context("row count overflow")?;
    ensure!(matrix.rows == total_rows && matrix.dimensions > 0 && matrix.rows.checked_mul(matrix.dimensions) == Some(matrix.data.len()), "invalid embedding matrix shape");
    ensure!(threshold.is_finite(), "threshold must be a finite f64");
    let mut unique = HashSet::new();
    for id in query_ids.iter().chain(gallery_ids) {
        validate_id(id)?;
        ensure!(unique.insert(id.as_str()), "query/gallery image IDs must be globally unique");
    }
    let split = query_ids.len() * matrix.dimensions;
    // Refine only the static gallery; preserve raw vectors in embeddings.npy.
    let refined = retrieval::gallery_dba(&matrix.data[split..], matrix.dimensions, 4, 2)?;
    let gallery = retrieval::Gallery::new(gallery_ids, &refined, matrix.dimensions)?;
    let results = matrix.data[..split].chunks_exact(matrix.dimensions)
        .map(|query| gallery.search(query, threshold)).collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(results.iter().all(|result| result.top10.len() == 10), "invalid top-ten size");
    fs::create_dir(output)?;
    let submission_path = output.join("submission.csv.partial");
    let candidates_path = output.join("candidates.csv.partial");
    let embeddings_path = output.join("embeddings.npy.partial");
    let mut submission = csv::WriterBuilder::new().has_headers(false).from_path(&submission_path)?;
    let mut candidates = csv::Writer::from_path(&candidates_path)?;
    candidates.write_record(["query_id", "gallery_id", "confidence"])?;
    for (qid, result) in query_ids.iter().zip(&results) {
        let mut row = Vec::with_capacity(11);
        row.push(qid.as_str());
        row.extend(result.top10.iter().map(|hit| hit.image_id));
        submission.write_record(row)?;
        if let Some(hit) = result.candidate {
            candidates.write_record([qid.as_str(), hit.image_id, &f64::from(hit.score).to_string()])?;
        }
    }
    submission.flush()?;
    candidates.flush()?;
    drop(submission);
    drop(candidates);
    write_npy(&embeddings_path, matrix)?;
    for name in ["submission.csv", "candidates.csv", "embeddings.npy"] {
        fs::rename(output.join(format!("{name}.partial")), output.join(name))?;
    }
    Ok(())
}

/// Resolve configuration-relative paths consistently across operating systems.
pub fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { base.join(path) }
}
