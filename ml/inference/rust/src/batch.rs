//! Bounded CPU-only preparation; CUDA sessions never enter worker closures.
use crate::{Config, Error, Input, Result, decode_crop, preprocess_crop};
use serde::{Deserialize, Serialize};
use std::thread::{Builder, ScopedJoinHandle};

/// Explicit, reversible CPU preprocessing choices. Defaults preserve the serial path.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct PreprocessingOptions {
    /// At most four workers; batch one always stays on the calling thread.
    pub workers: usize,
    /// Compute equal complete configurations once, then copy the resulting tensor.
    pub reuse_identical: bool,
}

impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            workers: 1,
            reuse_identical: false,
        }
    }
}

impl PreprocessingOptions {
    /// Reject unsupported worker counts rather than silently changing the option.
    pub fn validate(self) -> Result<()> {
        if !(1..=4).contains(&self.workers) {
            return Err(Error::Invalid(
                "preprocessing workers must be in 1..=4".into(),
            ));
        }
        Ok(())
    }
}

/// Contiguous float32 NCHW tensors, preserving original input row order.
pub struct PreparedBatch {
    /// First encoder input.
    pub first: Vec<f32>,
    /// Second encoder input, if a second configuration was provided.
    pub second: Option<Vec<f32>>,
}

fn prepare_one(
    input: &Input,
    first: &Config,
    second: Option<&Config>,
    reuse: bool,
) -> Result<PreparedBatch> {
    let crop = decode_crop(input)?;
    let tensor = preprocess_crop(&crop, first)?;
    let other = second
        .map(|config| {
            if reuse && first == config {
                Ok(tensor.clone())
            } else {
                preprocess_crop(&crop, config)
            }
        })
        .transpose()?;
    Ok(PreparedBatch {
        first: tensor,
        second: other,
    })
}

fn input_error(index: usize, error: Error) -> Error {
    Error::Invalid(format!("preprocessing input {index}: {error}"))
}

fn allocate(rows: usize, width: usize) -> Result<Vec<f32>> {
    let length = rows
        .checked_mul(width)
        .ok_or_else(|| Error::Invalid("batch tensor size overflow".into()))?;
    let mut tensor = Vec::new();
    tensor
        .try_reserve_exact(length)
        .map_err(|e| Error::Invalid(format!("batch allocation failed: {e}")))?;
    tensor.resize(length, 0.0);
    Ok(tensor)
}

type Worker<'a> = (usize, std::io::Result<ScopedJoinHandle<'a, Result<()>>>);

fn finish_workers(workers: Vec<Worker<'_>>) -> Result<()> {
    let mut first_error = None;
    for (first_index, worker) in workers {
        let result = match worker {
            Ok(handle) => match handle.join() {
                Ok(result) => result,
                Err(_) => Err(Error::Invalid(format!(
                    "preprocessing worker starting at input {first_index} panicked"
                ))),
            },
            Err(error) => Err(input_error(first_index, Error::Io(error))),
        };
        if first_error.is_none() {
            first_error = result.err();
        }
    }
    // Every worker is joined even after an earlier error, preventing scoped panic propagation.
    first_error.map_or(Ok(()), Err)
}

/// Prepare a batch without loading ORT. At most four images are decoded concurrently.
/// Output tensors are preallocated for parallel execution, with disjoint worker slices.
/// Errors follow original input/chunk order; all threads join before returning an error.
pub fn preprocess_batch(
    inputs: &[Input],
    first: &Config,
    second: Option<&Config>,
    options: PreprocessingOptions,
) -> Result<PreparedBatch> {
    options.validate()?;
    first.validate()?;
    if let Some(second) = second {
        second.validate()?;
    }
    if inputs.is_empty() {
        return Err(Error::Invalid("empty preprocessing batch".into()));
    }
    if options.workers == 1 || inputs.len() == 1 {
        let mut batch = PreparedBatch {
            first: Vec::new(),
            second: second.map(|_| Vec::new()),
        };
        for (index, input) in inputs.iter().enumerate() {
            let prepared = prepare_one(input, first, second, options.reuse_identical)
                .map_err(|e| input_error(index, e))?;
            batch.first.extend(prepared.first);
            if let (Some(output), Some(tensor)) = (&mut batch.second, prepared.second) {
                output.extend(tensor);
            }
        }
        return Ok(batch);
    }
    let first_width = 3 * (first.size as usize).pow(2);
    let second_width = second.map(|config| 3 * (config.size as usize).pow(2));
    let mut batch = PreparedBatch {
        first: allocate(inputs.len(), first_width)?,
        second: second_width
            .map(|width| allocate(inputs.len(), width))
            .transpose()?,
    };
    let chunk_rows = inputs.len().div_ceil(options.workers);
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        let mut second_chunks = batch
            .second
            .as_mut()
            .zip(second_width)
            .map(|(values, width)| values.chunks_mut(chunk_rows * width));
        for (chunk_index, (chunk, first_output)) in inputs
            .chunks(chunk_rows)
            .zip(batch.first.chunks_mut(chunk_rows * first_width))
            .enumerate()
        {
            let first_index = chunk_index * chunk_rows;
            let second_output = second_chunks.as_mut().and_then(Iterator::next);
            let handle = Builder::new()
                .name(format!("lct-preprocess-{chunk_index}"))
                .spawn_scoped(scope, move || {
                    let mut second_rows = second_output
                        .zip(second_width)
                        .map(|(output, width)| output.chunks_mut(width));
                    for (offset, (input, first_row)) in chunk
                        .iter()
                        .zip(first_output.chunks_mut(first_width))
                        .enumerate()
                    {
                        let prepared = prepare_one(input, first, second, options.reuse_identical)
                            .map_err(|error| input_error(first_index + offset, error))?;
                        first_row.copy_from_slice(&prepared.first);
                        if let (Some(output), Some(tensor)) = (
                            second_rows.as_mut().and_then(Iterator::next),
                            prepared.second,
                        ) {
                            output.copy_from_slice(&tensor);
                        }
                    }
                    Ok(())
                });
            workers.push((first_index, handle));
        }
        finish_workers(workers)
    })?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BBox, Mode, preprocess_pair};

    fn fixture(name: &str) -> (Input, Config) {
        let path =
            std::env::temp_dir().join(format!("lct-batch-{name}-{}.png", std::process::id()));
        image::RgbImage::from_fn(16, 18, |x, y| {
            image::Rgb([(x * 13) as u8, (y * 11) as u8, (x + y) as u8])
        })
        .save(&path)
        .unwrap();
        (
            Input {
                path,
                bbox: BBox {
                    x: 1,
                    y: 2,
                    w: 12,
                    h: 13,
                },
            },
            Config {
                size: 8,
                mean: [0.5; 3],
                std: [0.25; 3],
                mode: Mode::Center,
                crop_pct: 1.0,
            },
        )
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn unequal_complete_configs_are_never_reused_and_rows_stay_ordered() {
        let (input, first) = fixture("config");
        let mut alternatives = vec![first.clone(); 5];
        alternatives[0].size = 9;
        alternatives[1].mean[0] = 0.6;
        alternatives[2].std[1] = 0.4;
        alternatives[3].mode = Mode::Letterbox;
        alternatives[4].crop_pct = 0.75;
        let mut inputs = vec![input.clone(); 7];
        for (index, value) in inputs.iter_mut().enumerate() {
            value.bbox.w -= (index % 4) as i64;
        }
        for second in alternatives.iter().chain(std::iter::once(&first)) {
            let (mut expected_first, mut expected_second) = (Vec::new(), Vec::new());
            for value in &inputs {
                let (a, b) = preprocess_pair(value, &first, second).unwrap();
                expected_first.extend(a);
                expected_second.extend(b);
            }
            for workers in [1, 2, 3, 4] {
                let actual = preprocess_batch(
                    &inputs,
                    &first,
                    Some(second),
                    PreprocessingOptions {
                        workers,
                        reuse_identical: true,
                    },
                )
                .unwrap();
                assert_eq!(bits(&actual.first), bits(&expected_first));
                assert_eq!(bits(&actual.second.unwrap()), bits(&expected_second));
            }
        }
        std::fs::remove_file(input.path).unwrap();
    }

    #[test]
    fn invalid_inputs_return_earliest_original_index_across_worker_counts() {
        let (input, config) = fixture("errors");
        let mut inputs = vec![input.clone(); 8];
        inputs[2].bbox.x = -1;
        inputs[6].path = input.path.with_extension("missing");
        for workers in [1, 2, 3, 4] {
            for _ in 0..3 {
                let error = preprocess_batch(
                    &inputs,
                    &config,
                    Some(&config),
                    PreprocessingOptions {
                        workers,
                        reuse_identical: true,
                    },
                )
                .err()
                .unwrap();
                assert!(error.to_string().starts_with("preprocessing input 2:"));
            }
        }
        assert!(preprocess_batch(&[], &config, None, PreprocessingOptions::default()).is_err());
        std::fs::remove_file(input.path).unwrap();
    }

    #[test]
    fn single_encoder_and_batch_one_match_original_tensor_bits() {
        let (input, config) = fixture("single");
        let expected = crate::preprocess(&input, &config).unwrap();
        for count in [1, 7] {
            let actual = preprocess_batch(
                &vec![input.clone(); count],
                &config,
                None,
                PreprocessingOptions {
                    workers: 4,
                    reuse_identical: true,
                },
            )
            .unwrap();
            assert!(actual.second.is_none());
            for row in actual.first.chunks_exact(expected.len()) {
                assert_eq!(bits(row), bits(&expected));
            }
        }
        std::fs::remove_file(input.path).unwrap();
    }

    #[test]
    fn workers_reject_invalid_counts() {
        for workers in [0, 5, usize::MAX] {
            assert!(
                PreprocessingOptions {
                    workers,
                    reuse_identical: false
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn worker_panics_are_joined_and_earliest_chunk_error_wins() {
        let result = std::thread::scope(|scope| {
            finish_workers(vec![
                (
                    0,
                    Builder::new()
                        .spawn_scoped(scope, || Err(Error::Invalid("earlier input error".into()))),
                ),
                (
                    4,
                    Builder::new().spawn_scoped(scope, || panic!("deliberate CPU test panic")),
                ),
            ])
        });
        assert_eq!(result.err().unwrap().to_string(), "earlier input error");
        let result = std::thread::scope(|scope| {
            finish_workers(vec![(
                4,
                Builder::new().spawn_scoped(scope, || panic!("deliberate CPU test panic")),
            )])
        });
        assert_eq!(
            result.err().unwrap().to_string(),
            "preprocessing worker starting at input 4 panicked"
        );
    }
}
