//! CPU-only file-contract check; not the production image-inference CLI.
use anyhow::{Context, Result, ensure};
use lct_inference::Embeddings;
use lct_offline::{read_rows, write_submission};
use serde::Deserialize;
use std::{env, fs::{self, File}, path::PathBuf};

#[derive(Deserialize)]
struct Metadata { shape: [usize; 2] }

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    ensure!(args.len() == 6, "usage: check_saved META.json FEATURES.f32 QUERY.csv GALLERY.csv THRESHOLD OUTPUT");
    let meta: Metadata = serde_json::from_reader(File::open(&args[0])?)?;
    let bytes = fs::read(&args[1])?;
    let count = meta.shape[0].checked_mul(meta.shape[1]).context("shape overflow")?;
    ensure!(count.checked_mul(4) == Some(bytes.len()), "feature byte count mismatch");
    let data = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let matrix = Embeddings { rows: meta.shape[0], dimensions: meta.shape[1], data };
    let query = read_rows(&PathBuf::from(&args[2]))?;
    let gallery = read_rows(&PathBuf::from(&args[3]))?;
    let query_ids = query.into_iter().map(|r| r.image_id).collect::<Vec<_>>();
    let gallery_ids = gallery.into_iter().map(|r| r.image_id).collect::<Vec<_>>();
    write_submission(&query_ids, &gallery_ids, &matrix, args[4].parse()?, &PathBuf::from(&args[5]))
}
