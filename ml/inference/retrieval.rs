//! Independent cosine retrieval against an immutable gallery. No camera/identity input,
//! query-to-query state or backend dependencies. `gallery_dba` optionally refines the static gallery once.
//! Compile CPU tests with `rustc --edition=2024 --test inference/retrieval.rs -O -o retrieval-tests.exe`.
//! Production thresholds must be calibrated with this scorer and actual runtime vectors.

use std::{collections::HashSet, error::Error as StdError, fmt};

/// Maximum absolute deviation of the f64-measured L2 norm from one.
/// Accepted vectors are never modified, renormalized, or score-clamped.
pub const UNIT_NORM_TOLERANCE: f64 = 1e-5;

/// Invalid retrieval input; callers must not replace these errors with a prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Empty gallery, zero dimensions, overflow, or inconsistent matrix/query shape.
    Shape,
    /// Empty or duplicate opaque gallery image identifiers.
    GalleryId,
    /// Nonfinite values, zero norm, or norm outside the declared unit tolerance.
    Vector,
    /// Threshold is NaN or infinite.
    Threshold,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape => "invalid gallery or query shape",
            Self::GalleryId => "gallery image IDs must be unique and nonempty",
            Self::Vector => "expected a finite unit vector within norm tolerance 1e-5",
            Self::Threshold => "threshold must be finite",
        })
    }
}

impl StdError for Error {}

fn validate_vector(vector: &[f32]) -> Result<(), Error> {
    let squared_norm = vector
        .iter()
        .try_fold(0.0_f64, |sum, &value| {
            value.is_finite().then_some(sum + f64::from(value).powi(2))
        })
        .ok_or(Error::Vector)?;
    if squared_norm <= 0.0
        || !squared_norm.is_finite()
        || (squared_norm.sqrt() - 1.0).abs() > UNIT_NORM_TOLERANCE
    {
        return Err(Error::Vector);
    }
    Ok(())
}

/// An unfiltered gallery match. IDs identify images, never vehicle identity labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit<'a> {
    /// Original gallery row, also the deterministic tie breaker.
    pub gallery_index: usize,
    /// Borrowed gallery image identifier.
    pub image_id: &'a str,
    /// Sequential float32 dot product; no final clamp or re-normalization.
    pub score: f32,
}

/// Ranking and refusal decision are separate outputs.
#[derive(Debug, PartialEq)]
pub struct SearchResult<'a> {
    /// Up to ten matches, even when the candidate is refused.
    pub top10: Vec<Hit<'a>>,
    /// Top-1 only when its score is greater than or equal to the threshold.
    pub candidate: Option<Hit<'a>>,
}

/// Borrowed, validated row-major gallery; no mutable methods or cross-query state.
pub struct Gallery<'a> {
    image_ids: &'a [String],
    vectors: &'a [f32],
    dimensions: usize,
}

impl<'a> Gallery<'a> {
    /// Validate opaque IDs and pre-normalized float32 gallery vectors once.
    /// Returns an error for empty/duplicate IDs, invalid shapes or invalid vectors.
    pub fn new(
        image_ids: &'a [String],
        vectors: &'a [f32],
        dimensions: usize,
    ) -> Result<Self, Error> {
        if image_ids.is_empty()
            || dimensions == 0
            || image_ids.len().checked_mul(dimensions) != Some(vectors.len())
        {
            return Err(Error::Shape);
        }
        let mut unique = HashSet::with_capacity(image_ids.len());
        if image_ids
            .iter()
            .any(|id| id.is_empty() || !unique.insert(id.as_str()))
        {
            return Err(Error::GalleryId);
        }
        for vector in vectors.chunks_exact(dimensions) {
            validate_vector(vector)?;
        }
        Ok(Self {
            image_ids,
            vectors,
            dimensions,
        })
    }

    /// Score one independent pre-normalized query, in original gallery order.
    /// Accumulation is left-to-right f32 multiplication/addition, starting at +0.0;
    /// no explicit FMA, f64 sum, BLAS, parallel reduction or fast-math is used.
    /// Returns an error for wrong dimensions, nonfinite values or invalid norm.
    pub fn scores(&self, query: &[f32]) -> Result<Vec<f32>, Error> {
        if query.len() != self.dimensions {
            return Err(Error::Shape);
        }
        validate_vector(query)?;
        self.vectors
            .chunks_exact(self.dimensions)
            .map(|gallery| {
                let score = query
                    .iter()
                    .zip(gallery)
                    .fold(0.0_f32, |sum, (&q, &g)| sum + q * g);
                if score.is_finite() {
                    Ok(score)
                } else {
                    Err(Error::Vector)
                }
            })
            .collect()
    }

    /// Rank one query by descending score, breaking exact numeric ties by gallery
    /// row. Return raw top-10 regardless of refusal; acceptance uses inclusive `>=`.
    /// Returns an error for invalid query or nonfinite threshold.
    pub fn search(&self, query: &[f32], threshold: f64) -> Result<SearchResult<'a>, Error> {
        if !threshold.is_finite() {
            return Err(Error::Threshold);
        }
        let mut hits: Vec<_> = self
            .scores(query)?
            .into_iter()
            .enumerate()
            .map(|(index, score)| Hit {
                gallery_index: index,
                image_id: &self.image_ids[index],
                score,
            })
            .collect();
        // ponytail: sort the small gallery directly; optimize only after measuring a larger gallery.
        hits.sort_unstable_by(|a, b| {
            if a.score == b.score {
                a.gallery_index.cmp(&b.gallery_index)
            } else {
                b.score.total_cmp(&a.score)
            }
        });
        hits.truncate(10);
        let candidate = hits
            .first()
            .copied()
            .filter(|hit| f64::from(hit.score) >= threshold);
        Ok(SearchResult {
            top10: hits,
            candidate,
        })
    }
}

/// Gallery-side database augmentation (DBA), computed once over the static gallery only (allowed by Q38;
/// no query or query-to-query information). Each row becomes the L2-normalized sum of its top-(k+1) gallery
/// rows (itself included) weighted by max(cos, 0)^alpha. Ties keep gallery order. SEARCH dev: k=4, alpha=2.
pub fn gallery_dba(vectors: &[f32], dimensions: usize, k: usize, alpha: i32) -> Result<Vec<f32>, Error> {
    if dimensions == 0 || vectors.is_empty() || vectors.len() % dimensions != 0 {
        return Err(Error::Shape);
    }
    let rows: Vec<&[f32]> = vectors.chunks_exact(dimensions).collect();
    let mut out = Vec::with_capacity(vectors.len());
    for a in &rows {
        let mut sims: Vec<(f32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(j, b)| (a.iter().zip(*b).fold(0.0_f32, |s, (&x, &y)| s + x * y), j))
            .collect();
        sims.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        let mut acc = vec![0.0_f64; dimensions];
        for &(sim, j) in sims.iter().take(k + 1) {
            let w = f64::from(sim.max(0.0)).powi(alpha);
            for (t, &v) in acc.iter_mut().zip(rows[j]) {
                *t += w * f64::from(v);
            }
        }
        let norm = acc.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(Error::Vector);
        }
        out.extend(acc.iter().map(|v| (v / norm) as f32));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("image-{index}")).collect()
    }

    #[test]
    fn exact_ties_keep_gallery_order_and_refusal_keeps_raw_top_ten() {
        let ids = ids(12);
        let vectors = vec![1.0; 12];
        let gallery = Gallery::new(&ids, &vectors, 1).unwrap();
        let result = gallery.search(&[1.0], 1.01).unwrap();
        assert!(result.candidate.is_none());
        assert_eq!(
            result
                .top10
                .iter()
                .map(|hit| hit.gallery_index)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn equal_threshold_accepts_top_one_and_small_gallery_stays_small() {
        let ids = ids(2);
        let gallery = Gallery::new(&ids, &[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let result = gallery.search(&[0.0, 1.0], 1.0).unwrap();
        assert_eq!(result.candidate.unwrap().gallery_index, 1);
        assert_eq!(result.top10.len(), 2);
    }

    #[test]
    fn accepted_norm_tolerance_does_not_modify_or_normalize_vectors() {
        let ids = ids(1);
        let near_unit = 1.0_f32 + 2e-6;
        let vectors = [near_unit, 0.0];
        let gallery = Gallery::new(&ids, &vectors, 2).unwrap();
        assert_eq!(
            gallery.scores(&vectors).unwrap(),
            vec![near_unit * near_unit]
        );
        assert_eq!(vectors[0], near_unit);
        assert!(gallery.scores(&vectors).unwrap()[0] > 1.0);
    }

    #[test]
    fn query_order_has_no_effect_and_scores_keep_gallery_order() {
        let ids = ids(2);
        let gallery = Gallery::new(&ids, &[1.0, 0.0, -1.0, 0.0], 2).unwrap();
        let before = gallery.search(&[1.0, 0.0], 0.0).unwrap();
        assert_eq!(gallery.scores(&[-1.0, 0.0]).unwrap(), vec![-1.0, 1.0]);
        assert_eq!(before, gallery.search(&[1.0, 0.0], 0.0).unwrap());
    }

    #[test]
    fn malformed_gallery_is_rejected() {
        let ids = ids(1);
        assert!(matches!(Gallery::new(&ids, &[1.0], 0), Err(Error::Shape)));
        assert!(matches!(
            Gallery::new(&ids, &[1.0], usize::MAX),
            Err(Error::Shape)
        ));
        assert!(matches!(Gallery::new(&[], &[], 1), Err(Error::Shape)));
        assert!(matches!(
            Gallery::new(&["".into()], &[1.0], 1),
            Err(Error::GalleryId)
        ));
        assert!(matches!(
            Gallery::new(&["x".into(), "x".into()], &[1.0, 1.0], 1),
            Err(Error::GalleryId)
        ));
        for vector in [
            [0.0, 0.0],
            [1.001, 0.0],
            [f32::NAN, 0.0],
            [f32::INFINITY, 0.0],
        ] {
            assert!(matches!(Gallery::new(&ids, &vector, 2), Err(Error::Vector)));
        }
    }

    #[test]
    fn malformed_query_and_threshold_are_rejected() {
        let ids = ids(1);
        let gallery = Gallery::new(&ids, &[1.0, 0.0], 2).unwrap();
        assert_eq!(gallery.scores(&[1.0]), Err(Error::Shape));
        for vector in [
            [0.0, 0.0],
            [1.001, 0.0],
            [f32::NAN, 0.0],
            [f32::NEG_INFINITY, 0.0],
        ] {
            assert_eq!(gallery.scores(&vector), Err(Error::Vector));
        }
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                gallery.search(&[1.0, 0.0], threshold),
                Err(Error::Threshold)
            );
        }
    }

    #[test]
    fn f64_reject_all_boundary_above_f32_score_is_not_rounded_down() {
        let ids = ids(1);
        let gallery = Gallery::new(&ids, &[1.0], 1).unwrap();
        let just_above_one = f64::from_bits(1.0_f64.to_bits() + 1);
        assert!(
            gallery
                .search(&[1.0], just_above_one)
                .unwrap()
                .candidate
                .is_none()
        );
    }

    #[test]
    fn dba_pools_top_neighbours_and_stays_unit_norm() {
        let v = [1.0, 0.0, 0.8, 0.6, 0.0, 1.0];
        let out = gallery_dba(&v, 2, 1, 2).unwrap();
        for row in out.chunks_exact(2) {
            let n: f32 = row.iter().map(|x| x * x).sum();
            assert!((n - 1.0).abs() < 1e-5);
        }
        // row 0 pools itself (w=1) and row 1 (w=0.64): (1.512, 0.384) normalized.
        assert!((out[0] - 0.96923).abs() < 1e-4 && (out[1] - 0.24615).abs() < 1e-4);
    }
}
