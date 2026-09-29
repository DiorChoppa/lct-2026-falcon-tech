//! Offline vehicle image preprocessing and a resident CUDA embedding extractor.

use std::path::{Path, PathBuf};

mod batch;
mod jpeg;
mod prefetch;
mod metadata;
mod resize;

pub use batch::{PreparedBatch, PreprocessingOptions, preprocess_batch};

use image::{Rgb, RgbImage, imageops};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use serde::{Deserialize, Serialize};

/// Errors from input validation, image decoding, and the native runtime.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The configuration, image bounds, or model output is invalid.
    #[error("{0}")]
    Invalid(String),
    /// Image reading or decoding failed.
    #[error(transparent)]
    Image(#[from] image::ImageError),
    /// Reading the original encoded file failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// libjpeg-turbo could not decode the JPEG.
    #[error(transparent)]
    Jpeg(#[from] turbojpeg::Error),
    /// CUDA runtime loading or execution failed.
    #[error(transparent)]
    Runtime(#[from] ort::Error),
}

/// Operations return a diagnostic instead of changing device or input policy.
pub type Result<T> = std::result::Result<T, Error>;

/// How to map the complete bbox crop into the square network input.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Resize the short edge, then center crop with ties-to-even offsets.
    Center,
    /// Resize directly to a square.
    Stretch,
    /// Fit the long edge, preserving aspect ratio and filling with rounded mean RGB.
    Letterbox,
}

/// The complete preprocessing specification; values must match the trained export.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Config {
    /// Square model input dimension.
    pub size: u32,
    /// RGB normalization mean.
    pub mean: [f64; 3],
    /// RGB normalization standard deviation.
    pub std: [f64; 3],
    /// Geometric transformation.
    pub mode: Mode,
    /// Center-mode short-edge resize is floor(size / crop_pct).
    #[serde(default = "one")]
    pub crop_pct: f64,
}

fn one() -> f64 {
    1.0
}

impl Config {
    /// Reject malformed values before allocating image tensors.
    pub fn validate(&self) -> Result<()> {
        if self.size == 0 || self.size > 4096 {
            return Err(Error::Invalid("size must be between 1 and 4096".into()));
        }
        if !self.crop_pct.is_finite()
            || self.crop_pct <= 0.0
            || self.crop_pct > 1.0
            || f64::from(self.size) / self.crop_pct > 4096.0
        {
            return Err(Error::Invalid(
                "crop_pct must yield a short edge between size and 4096".into(),
            ));
        }
        if self
            .mean
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || self
                .std
                .iter()
                .any(|x| !(*x as f32).is_finite() || *x as f32 <= 0.0)
        {
            return Err(Error::Invalid(
                "mean must be finite in [0,1]; std must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// Integer XYWH bounds in the original frame, with no silent clipping.
#[derive(Clone, Copy, Debug)]
pub struct BBox {
    /// Left coordinate.
    pub x: i64,
    /// Top coordinate.
    pub y: i64,
    /// Width.
    pub w: i64,
    /// Height.
    pub h: i64,
}

/// One observation. Its position in a batch is preserved in the output.
#[derive(Clone, Debug)]
pub struct Input {
    /// Path to the original encoded image.
    pub path: PathBuf,
    /// Target vehicle bounds.
    pub bbox: BBox,
}

fn checked_bbox(bbox: BBox, width: u32, height: u32) -> Result<(u32, u32, u32, u32)> {
    if bbox.x < 0
        || bbox.y < 0
        || bbox.w <= 0
        || bbox.h <= 0
        || bbox
            .x
            .checked_add(bbox.w)
            .is_none_or(|right| right > i64::from(width))
        || bbox
            .y
            .checked_add(bbox.h)
            .is_none_or(|bottom| bottom > i64::from(height))
    {
        return Err(Error::Invalid(format!(
            "invalid bbox {bbox:?} for {width}x{height}"
        )));
    }
    Ok((bbox.x as u32, bbox.y as u32, bbox.w as u32, bbox.h as u32))
}

fn center_offset(gap: u32) -> u32 {
    let half = gap / 2;
    half + u32::from(gap % 2 == 1 && half % 2 == 1)
}

/// Produce square RGB pixels after geometry, before float normalization.
pub fn transform_rgb(crop: &RgbImage, config: &Config) -> Result<RgbImage> {
    config.validate()?;
    let (width, height) = crop.dimensions();
    if width == 0 || height == 0 {
        return Err(Error::Invalid("empty crop".into()));
    }
    let size = config.size;
    let resize = |w, h| resize::bicubic(crop, w, h);
    Ok(match config.mode {
        Mode::Stretch => resize(size, size),
        Mode::Center => {
            let short = (f64::from(size) / config.crop_pct).floor() as u32;
            let (w, h) = if width <= height {
                (
                    short,
                    (u64::from(short) * u64::from(height) / u64::from(width)) as u32,
                )
            } else {
                (
                    (u64::from(short) * u64::from(width) / u64::from(height)) as u32,
                    short,
                )
            };
            if w > 65536 || h > 65536 {
                return Err(Error::Invalid(
                    "center resize would exceed 65536 pixels per edge".into(),
                ));
            }
            let resized = resize(w, h);
            imageops::crop_imm(
                &resized,
                center_offset(w - size),
                center_offset(h - size),
                size,
                size,
            )
            .to_image()
        }
        Mode::Letterbox => {
            let scale = f64::from(size) / f64::from(width.max(height));
            let w = (f64::from(width) * scale).round_ties_even().max(1.0) as u32;
            let h = (f64::from(height) * scale).round_ties_even().max(1.0) as u32;
            let fill = Rgb(config
                .mean
                .map(|value| (value * 255.0).round_ties_even() as u8));
            let mut result = RgbImage::from_pixel(size, size, fill);
            imageops::replace(
                &mut result,
                &resize(w, h),
                i64::from((size - w) / 2),
                i64::from((size - h) / 2),
            );
            result
        }
    })
}

#[cfg(test)]
fn decode_rgb(path: &Path) -> Result<RgbImage> {
    let bytes = std::fs::read(path)?;
    if bytes.starts_with(&[0xff, 0xd8]) {
        // Match Pillow's accurate libjpeg-turbo decode and smooth chroma upsampling.
        let decoded = turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGB)?;
        let width = u32::try_from(decoded.width)
            .map_err(|_| Error::Invalid("JPEG width exceeds u32".into()))?;
        let height = u32::try_from(decoded.height)
            .map_err(|_| Error::Invalid("JPEG height exceeds u32".into()))?;
        RgbImage::from_raw(width, height, decoded.pixels)
            .ok_or_else(|| Error::Invalid("Invalid decoded RGB buffer size".into()))
    } else {
        Ok(image::load_from_memory(&bytes)?.into_rgb8())
    }
}

fn decode_crop(input: &Input) -> Result<RgbImage> {
    let bytes = std::fs::read(&input.path)?;
    if bytes.starts_with(&[0xff, 0xd8]) {
        return jpeg::crop(&bytes, input.bbox);
    }
    let frame = image::load_from_memory(&bytes)?.into_rgb8();
    let (x, y, w, h) = checked_bbox(input.bbox, frame.width(), frame.height())?;
    Ok(imageops::crop_imm(&frame, x, y, w, h).to_image())
}

fn preprocess_crop(crop: &RgbImage, config: &Config) -> Result<Vec<f32>> {
    let square = transform_rgb(crop, config)?;
    let plane = (u64::from(config.size) * u64::from(config.size)) as usize;
    let mut output = vec![0.0; plane * 3];
    for (pixel_index, pixel) in square.pixels().enumerate() {
        for channel in 0..3 {
            output[channel * plane + pixel_index] = (f32::from(pixel[channel]) / 255.0
                - config.mean[channel] as f32)
                / config.std[channel] as f32;
        }
    }
    Ok(output)
}

/// Read/decode/crop/preprocess an original image into contiguous float32 CHW.
pub fn preprocess(input: &Input, config: &Config) -> Result<Vec<f32>> {
    config.validate()?;
    preprocess_crop(&decode_crop(input)?, config)
}

/// Decode and crop once, then independently transform the shared RGB crop.
pub fn preprocess_pair(
    input: &Input,
    first: &Config,
    second: &Config,
) -> Result<(Vec<f32>, Vec<f32>)> {
    first.validate()?;
    second.validate()?;
    let crop = decode_crop(input)?;
    Ok((
        preprocess_crop(&crop, first)?,
        preprocess_crop(&crop, second)?,
    ))
}

/// A contiguous row-major embedding matrix.
#[derive(Debug)]
pub struct Embeddings {
    /// Number of observations.
    pub rows: usize,
    /// Embedding width.
    pub dimensions: usize,
    /// Normalized float32 values in input-row order.
    pub data: Vec<f32>,
}

struct Encoder {
    session: Session,
    config: Config,
    policy_evidence: Option<serde_json::Value>,
}

impl Encoder {
    fn load(
        model: &Path,
        runtime: &Path,
        config: Config,
        profile_suffix: &str,
        policy_path: Option<&Path>,
    ) -> Result<Self> {
        config.validate()?;
        let policy = policy_path
            .map(|path| metadata::Policy::load(path, model, runtime, &config))
            .transpose()?;
        if policy.is_some() && std::env::var_os("ORT_PROFILE_PREFIX").is_some() {
            return Err(Error::Invalid("metadata policy uses startup profiling; unset ORT_PROFILE_PREFIX and use ORT_POLICY_PROFILE_DIR".into()));
        }
        let audit_directory = if policy.is_some() {
            Some(metadata::audit_directory()?)
        } else {
            None
        };
        let cuda = ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .build()
            .error_on_failure();
        let mut builder = if let Some(directory) = &audit_directory {
            Session::builder()?
                .with_execution_providers([cuda, ep::CPU::default().build().error_on_failure()])
                .map_err(ort::Error::<()>::from)?
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(ort::Error::<()>::from)?
                .with_profiling(directory.join("startup"))
                .map_err(ort::Error::<()>::from)?
        } else {
            Session::builder()?
                .with_execution_providers([cuda])
                .map_err(ort::Error::<()>::from)?
                .with_disable_cpu_fallback()
                .map_err(ort::Error::<()>::from)?
        };
        if let Some(mut prefix) = std::env::var_os("ORT_PROFILE_PREFIX") {
            prefix.push(profile_suffix);
            builder = builder
                .with_profiling(prefix)
                .map_err(ort::Error::<()>::from)?;
        }
        let session = builder.commit_from_file(model)?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err(Error::Invalid(
                "expected one NCHW input and one embedding output".into(),
            ));
        }
        let mut encoder = Self {
            session,
            config,
            policy_evidence: None,
        };
        if let (Some(policy), Some(policy_path), Some(directory)) =
            (policy, policy_path, audit_directory)
        {
            let warmup = encoder.forward(
                vec![0.0; 3 * encoder.config.size as usize * encoder.config.size as usize],
                1,
            );
            // End the same session's profiling even when numerical validation fails.
            let profile = encoder.session.end_profiling();
            let result: Result<serde_json::Value> = (|| {
                let warmup = warmup?;
                if warmup.dimensions != policy.embedding_dimension {
                    return Err(Error::Invalid(
                        "metadata policy output dimension mismatch".into(),
                    ));
                }
                let profile = profile
                    .as_ref()
                    .map_err(|error| Error::Invalid(error.to_string()))?;
                let mut report = policy.verify_profile(Path::new(profile))?;
                report["policy_path"] = serde_json::json!(policy_path);
                report["policy_sha256"] = serde_json::json!(metadata::sha256(policy_path)?);
                report["audit_directory"] = serde_json::json!(directory);
                report["startup_input"] = serde_json::json!("all zeros, float32");
                report["startup_profile_ended_before_serving"] = serde_json::json!(true);
                Ok(report)
            })();
            let report = match &result {
                Ok(report) => report.clone(),
                Err(error) => serde_json::json!({"status":"rejected", "error":error.to_string(),
                    "policy_path":policy_path,"profile":profile.as_ref().ok(),"audit_directory":directory}),
            };
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join("audit.json"))?;
            serde_json::to_writer_pretty(file, &report)
                .map_err(|e| Error::Invalid(e.to_string()))?;
            encoder.policy_evidence = Some(result?);
        }
        Ok(encoder)
    }

    fn forward(&mut self, batch: Vec<f32>, rows: usize) -> Result<Embeddings> {
        let shape = [
            rows,
            3,
            self.config.size as usize,
            self.config.size as usize,
        ];
        let tensor = Tensor::from_array((shape, batch))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 2 || shape[0] != rows as i64 || shape[1] <= 0 {
            return Err(Error::Invalid(format!(
                "expected [N,D] output, got {shape:?}"
            )));
        }
        let dimensions = shape[1] as usize;
        let mut data = values.to_vec();
        normalize_rows(&mut data, dimensions)?;
        Ok(Embeddings {
            rows,
            dimensions,
            data,
        })
    }
}

fn normalize_rows(data: &mut [f32], dimensions: usize) -> Result<()> {
    for row in data.chunks_exact_mut(dimensions) {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 1e-12 {
            return Err(Error::Invalid(
                "embedding is nonfinite or has zero norm".into(),
            ));
        }
        for value in row {
            *value /= norm;
        }
    }
    Ok(())
}

fn validate_weight(weight: f32) -> Result<()> {
    if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
        return Err(Error::Invalid(
            "fusion weight must be finite in [0,1]".into(),
        ));
    }
    Ok(())
}

/// Normalize both matrices, concatenate sqrt-weighted rows, then normalize again.
/// `first_weight` is the first encoder's cosine weight; the second gets `1-weight`.
pub fn weighted_concat(
    mut first: Embeddings,
    mut second: Embeddings,
    first_weight: f32,
) -> Result<Embeddings> {
    validate_weight(first_weight)?;
    for matrix in [&first, &second] {
        if matrix.rows == 0
            || matrix.dimensions == 0
            || matrix.rows.checked_mul(matrix.dimensions) != Some(matrix.data.len())
        {
            return Err(Error::Invalid("invalid embedding matrix shape".into()));
        }
    }
    if first.rows != second.rows {
        return Err(Error::Invalid("encoder row counts differ".into()));
    }
    let dimensions = first
        .dimensions
        .checked_add(second.dimensions)
        .ok_or_else(|| Error::Invalid("fused dimensions overflow".into()))?;
    let length = first
        .rows
        .checked_mul(dimensions)
        .ok_or_else(|| Error::Invalid("fused matrix size overflow".into()))?;
    normalize_rows(&mut first.data, first.dimensions)?;
    normalize_rows(&mut second.data, second.dimensions)?;
    let mut data = Vec::with_capacity(length);
    let first_scale = first_weight.sqrt();
    let second_scale = (1.0 - first_weight).sqrt();
    for (left, right) in first
        .data
        .chunks_exact(first.dimensions)
        .zip(second.data.chunks_exact(second.dimensions))
    {
        data.extend(left.iter().map(|value| value * first_scale));
        data.extend(right.iter().map(|value| value * second_scale));
    }
    normalize_rows(&mut data, dimensions)?;
    Ok(Embeddings {
        rows: first.rows,
        dimensions,
        data,
    })
}

/// One or two resident CUDA encoders, sharing image decode/crop when paired.
pub struct Extractor {
    first: Encoder,
    second: Option<Encoder>,
    first_weight: f32,
    runtime: PathBuf,
    preprocessing: PreprocessingOptions,
}

impl Extractor {
    /// Load a pinned native runtime and one ONNX graph; fail if CUDA cannot execute it.
    pub fn load(model: &Path, runtime: &Path, config: Config) -> Result<Self> {
        Self::load_internal(model, runtime, config, None)
    }

    /// Explicitly permit only graph-bound, profiled host metadata nodes at startup.
    pub fn load_with_metadata_policy(
        model: &Path,
        runtime: &Path,
        config: Config,
        policy: &Path,
    ) -> Result<Self> {
        Self::load_internal(model, runtime, config, Some(policy))
    }

    fn load_internal(
        model: &Path,
        runtime: &Path,
        config: Config,
        policy: Option<&Path>,
    ) -> Result<Self> {
        config.validate()?;
        let runtime = runtime.canonicalize()?;
        ort::init_from(&runtime)?.commit();
        Ok(Self {
            first: Encoder::load(model, &runtime, config, "", policy)?,
            second: None,
            first_weight: 1.0,
            runtime,
            preprocessing: PreprocessingOptions::default(),
        })
    }

    /// Add a second resident CUDA encoder; `first_weight` weights the original model.
    /// Repeated calls are rejected. Both encoders use device0 and fail on CPU fallback.
    pub fn with_second_model(
        self,
        model: &Path,
        config: Config,
        first_weight: f32,
    ) -> Result<Self> {
        self.add_second(model, config, first_weight, None)
    }

    /// Add an explicitly audited host-metadata second encoder; first stays unchanged.
    pub fn with_second_model_and_metadata_policy(
        self,
        model: &Path,
        config: Config,
        first_weight: f32,
        policy: &Path,
    ) -> Result<Self> {
        self.add_second(model, config, first_weight, Some(policy))
    }

    fn add_second(
        mut self,
        model: &Path,
        config: Config,
        first_weight: f32,
        policy: Option<&Path>,
    ) -> Result<Self> {
        validate_weight(first_weight)?;
        if self.second.is_some() {
            return Err(Error::Invalid(
                "second encoder is already configured".into(),
            ));
        }
        self.second = Some(Encoder::load(
            model,
            &self.runtime,
            config,
            "-second",
            policy,
        )?);
        self.first_weight = first_weight;
        Ok(self)
    }

    /// Startup placement evidence; null components retain strict CUDA-only policy.
    pub fn runtime_policy_evidence(&self) -> serde_json::Value {
        serde_json::json!({"first":self.first.policy_evidence,
            "second":self.second.as_ref().and_then(|second| second.policy_evidence.as_ref())})
    }

    /// Set explicit CPU preparation choices; defaults are one worker and no reuse.
    pub fn with_preprocessing_options(mut self, options: PreprocessingOptions) -> Result<Self> {
        options.validate()?;
        self.preprocessing = options;
        Ok(self)
    }

    /// Finish profiling; paired profiles are returned as two newline-separated paths.
    pub fn end_profiling(&mut self) -> Result<String> {
        let mut paths = self.first.session.end_profiling()?;
        if let Some(second) = &mut self.second {
            paths.push('\n');
            paths.push_str(&second.session.end_profiling()?);
        }
        Ok(paths)
    }

    /// Include decode/crop, independent transforms, sequential CUDA forwards and L2/fusion.
    pub fn extract_batch(&mut self, inputs: &[Input]) -> Result<Embeddings> {
        if inputs.is_empty() {
            return Err(Error::Invalid("empty extraction batch".into()));
        }
        let prepared = preprocess_batch(
            inputs,
            &self.first.config,
            self.second.as_ref().map(|second| &second.config),
            self.preprocessing,
        )?;
        self.encode_prepared(prepared, inputs.len())
    }

    fn encode_prepared(&mut self, prepared: PreparedBatch, rows: usize) -> Result<Embeddings> {
        if let Some(second) = &mut self.second {
            let second_batch = prepared
                .second
                .ok_or_else(|| Error::Invalid("missing second input tensor".into()))?;
            let first = self.first.forward(prepared.first, rows)?;
            let second = second.forward(second_batch, rows)?;
            weighted_concat(first, second, self.first_weight)
        } else {
            self.first.forward(prepared.first, rows)
        }
    }

    /// Ordered batches with one CPU-prepared future batch for sizes 2..=32.
    /// Batch one and larger legacy batches remain synchronous. Sessions stay here.
    pub fn extract_batches<I, C>(&mut self, batches: I, batch_size: usize, mut consume: C) -> Result<()>
    where
        I: Iterator + Send,
        I::Item: AsRef<[Input]>,
        C: FnMut(Embeddings) -> Result<bool>,
    {
        if batch_size == 0 { return Err(Error::Invalid("zero extraction batch size".into())); }
        if batch_size == 1 || batch_size > 32 {
            for batch in batches {
                let inputs = batch.as_ref();
                if inputs.is_empty() || inputs.len() > batch_size { return Err(Error::Invalid("invalid batch row count".into())); }
                if !consume(self.extract_batch(inputs)?)? { break; }
            }
            return Ok(());
        }
        let first = self.first.config.clone();
        let second = self.second.as_ref().map(|encoder| encoder.config.clone());
        let options = self.preprocessing;
        let prepared = batches.map(move |batch| {
            let inputs = batch.as_ref();
            let rows = inputs.len();
            if rows == 0 || rows > batch_size { return Err(Error::Invalid("invalid batch row count".into())); }
            Ok((rows, preprocess_batch(inputs, &first, second.as_ref(), options)?))
        });
        prefetch::ordered(prepared, |(rows, prepared)| consume(self.encode_prepared(prepared, rows)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_preprocessing_matches_independent_224_and_256_transforms() {
        let path = std::env::temp_dir().join(format!("lct-pair-{}.png", std::process::id()));
        let frame = RgbImage::from_fn(32, 48, |x, y| {
            Rgb([(x * 7) as u8, (y * 5) as u8, (x + y) as u8])
        });
        frame.save(&path).unwrap();
        let input = Input {
            path: path.clone(),
            bbox: BBox {
                x: 5,
                y: 7,
                w: 17,
                h: 31,
            },
        };
        let first = Config {
            size: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            mode: Mode::Stretch,
            crop_pct: 1.0,
        };
        let second = Config {
            size: 256,
            mean: [0.5; 3],
            std: [0.5; 3],
            mode: Mode::Stretch,
            crop_pct: 1.0,
        };
        let pair = preprocess_pair(&input, &first, &second).unwrap();
        let expected = (
            preprocess(&input, &first).unwrap(),
            preprocess(&input, &second).unwrap(),
        );
        std::fs::remove_file(path).unwrap();
        assert_eq!((pair.0.len(), pair.1.len()), (3 * 224 * 224, 3 * 256 * 256));
        assert_eq!(pair, expected);
    }

    #[test]
    fn fused_768_and_384_vectors_are_unit_norm_and_preserve_row_order() {
        let mut first = vec![0.0; 2 * 768];
        let mut second = vec![0.0; 2 * 384];
        first[0] = 2.0;
        first[768 + 1] = 3.0;
        second[2] = 4.0;
        second[384 + 3] = 5.0;
        let fused = weighted_concat(
            Embeddings {
                rows: 2,
                dimensions: 768,
                data: first,
            },
            Embeddings {
                rows: 2,
                dimensions: 384,
                data: second,
            },
            0.5,
        )
        .unwrap();
        assert_eq!(
            (fused.rows, fused.dimensions, fused.data.len()),
            (2, 1152, 2304)
        );
        for row in fused.data.chunks_exact(1152) {
            assert!((row.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
        }
        for index in [0, 770, 1153, 1923] {
            assert!((fused.data[index] - 0.5_f32.sqrt()).abs() < 1e-6);
        }
        assert_eq!(fused.data.iter().filter(|&&value| value != 0.0).count(), 4);
    }

    #[test]
    fn fused_cosine_is_weighted_component_cosine() {
        let fused = weighted_concat(
            Embeddings {
                rows: 2,
                dimensions: 2,
                data: vec![1.0, 0.0, 0.6, 0.8],
            },
            Embeddings {
                rows: 2,
                dimensions: 3,
                data: vec![0.0, 1.0, 0.0, 0.0, -0.8, 0.6],
            },
            0.3,
        )
        .unwrap();
        let cosine = fused.data[..5]
            .iter()
            .zip(&fused.data[5..])
            .map(|(a, b)| a * b)
            .sum::<f32>();
        assert!((cosine - (0.3 * 0.6 + 0.7 * -0.8)).abs() < 1e-6);
    }

    #[test]
    fn fusion_rejects_invalid_weights_shapes_and_nonfinite_vectors() {
        let valid = || Embeddings {
            rows: 1,
            dimensions: 2,
            data: vec![1.0, 0.0],
        };
        for weight in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            assert!(weighted_concat(valid(), valid(), weight).is_err());
        }
        for invalid in [
            Embeddings {
                rows: 2,
                dimensions: 2,
                data: vec![1.0, 0.0, 1.0, 0.0],
            },
            Embeddings {
                rows: 1,
                dimensions: 3,
                data: vec![1.0, 0.0],
            },
            Embeddings {
                rows: 1,
                dimensions: 2,
                data: vec![0.0, 0.0],
            },
            Embeddings {
                rows: 1,
                dimensions: 2,
                data: vec![f32::NAN, 0.0],
            },
        ] {
            assert!(weighted_concat(valid(), invalid, 0.5).is_err());
        }
        for weight in [0.0, 1.0] {
            let fused = weighted_concat(valid(), valid(), weight).unwrap();
            assert_eq!(fused.data.iter().map(|v| v * v).sum::<f32>(), 1.0);
        }
    }

    #[test]
    #[ignore = "requires the local fitting-only Pillow crop fixture"]
    fn jpeg_fit_fixture_crops_match_pillow_rgb_bytes() {
        #[derive(Deserialize)]
        struct FixtureRow {
            image_id: String,
            path: PathBuf,
            x: u32,
            y: u32,
            w: u32,
            h: u32,
        }
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/inference_preprocessing");
        let mut manifest = csv::Reader::from_path(root.join("manifest_jpeg.csv")).unwrap();
        let mut count = 0;
        for row in manifest.deserialize::<FixtureRow>() {
            let row = row.unwrap();
            let decoded = decode_rgb(&row.path).unwrap();
            let crop = imageops::crop_imm(&decoded, row.x, row.y, row.w, row.h).to_image();
            let reference = image::open(
                root.join("pillow_decoded_crops")
                    .join(format!("{}.png", row.image_id)),
            )
            .unwrap()
            .into_rgb8();
            assert!(crop == reference, "JPEG RGB crop differs: {}", row.image_id);
            count += 1;
        }
        assert_eq!(count, 128);
    }

    #[test]
    fn center_crop_offsets_match_python_ties_to_even() {
        assert_eq!(
            (0..8).map(center_offset).collect::<Vec<_>>(),
            vec![0, 0, 1, 2, 2, 2, 3, 4]
        );
    }

    #[test]
    fn bbox_rejects_overflow_and_outside_frame() {
        for bbox in [
            BBox {
                x: i64::MAX,
                y: 0,
                w: 2,
                h: 1,
            },
            BBox {
                x: -1,
                y: 0,
                w: 1,
                h: 1,
            },
            BBox {
                x: 9,
                y: 0,
                w: 2,
                h: 1,
            },
        ] {
            assert!(checked_bbox(bbox, 10, 10).is_err());
        }
    }

    #[test]
    fn letterbox_places_one_pixel_wide_crop_with_mean_padding() {
        let config = Config {
            size: 4,
            mean: [0.5; 3],
            std: [1.0; 3],
            mode: Mode::Letterbox,
            crop_pct: 1.0,
        };
        let result = transform_rgb(&RgbImage::from_pixel(1, 4, Rgb([255, 0, 0])), &config).unwrap();
        assert_eq!(result.get_pixel(0, 0).0, [128; 3]);
        assert_eq!(result.get_pixel(1, 0).0, [255, 0, 0]);
    }

    #[test]
    fn letterbox_fill_uses_python_double_precision_rounding() {
        let config = Config {
            size: 4,
            mean: [0.3, 0.5, 0.7],
            std: [1.0; 3],
            mode: Mode::Letterbox,
            crop_pct: 1.0,
        };
        let result = transform_rgb(&RgbImage::from_pixel(1, 4, Rgb([0; 3])), &config).unwrap();
        assert_eq!(result.get_pixel(0, 0).0, [76, 128, 178]);
    }
}


#[cfg(test)]
mod prefetched_tensor_tests {
    use super::*;
    #[test]
    fn ordered_prefetch_preserves_prepared_tensor_bits() {
        let path = std::env::temp_dir().join(format!("lct-prefetch-parity-{}.png", std::process::id()));
        image::RgbImage::from_fn(16, 18, |x,y| image::Rgb([(x*13) as u8, (y*11) as u8, (x+y) as u8])).save(&path).unwrap();
        let rows: Vec<Input> = (0..7).map(|i| Input { path: path.clone(), bbox: BBox { x:i%3, y:1, w:12-i%3, h:13 } }).collect();
        let first = Config { size:8, mean:[0.5;3], std:[0.25;3], mode:Mode::Stretch, crop_pct:1.0 };
        let second = Config { size:9, ..first.clone() };
        let options = PreprocessingOptions { workers:4, reuse_identical:true };
        let bits = |p: PreparedBatch| (p.first.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), p.second.unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let expected: Vec<_> = rows.chunks(3).map(|chunk| bits(preprocess_batch(chunk, &first, Some(&second), options).unwrap())).collect();
        let mut observed = Vec::new();
        prefetch::ordered(rows.chunks(3).map(|chunk| preprocess_batch(chunk, &first, Some(&second), options)), |p| { observed.push(bits(p)); Ok(true) }).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(observed, expected);
    }
}
