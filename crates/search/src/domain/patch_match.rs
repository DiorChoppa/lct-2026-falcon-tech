#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchMatch {
    pub query_index: usize,
    pub candidate_index: usize,
    pub similarity: f32,
}

/// Mutual-nearest-neighbor matching between two sets of L2-normalized patch
/// tokens (flat `[n_tokens * patch_dim]`, as returned by inference's
/// `with_patches` — each token already L2-normalized, so plain dot product
/// is cosine similarity). Returns matched pairs and the aggregate
/// local_score: summed similarity of mutual pairs, normalized by the
/// smaller token-set size, clamped to 0..1.
pub fn match_patches(query: &[f32], candidate: &[f32], patch_dim: usize) -> (Vec<PatchMatch>, f32) {
    if patch_dim == 0 {
        return (vec![], 0.0);
    }
    let query_tokens: Vec<&[f32]> = query.chunks_exact(patch_dim).collect();
    let candidate_tokens: Vec<&[f32]> = candidate.chunks_exact(patch_dim).collect();
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return (vec![], 0.0);
    }

    let query_best: Vec<(usize, f32)> = query_tokens
        .iter()
        .map(|q| best_match(q, &candidate_tokens))
        .collect();
    let candidate_best: Vec<(usize, f32)> = candidate_tokens
        .iter()
        .map(|c| best_match(c, &query_tokens))
        .collect();

    let matches: Vec<PatchMatch> = query_best
        .iter()
        .enumerate()
        .filter(|(qi, (ci, _))| candidate_best[*ci].0 == *qi)
        .map(|(qi, (ci, sim))| PatchMatch {
            query_index: qi,
            candidate_index: *ci,
            similarity: *sim,
        })
        .collect();

    let denom = query_tokens.len().min(candidate_tokens.len()) as f32;
    let local_score =
        (matches.iter().map(|m| m.similarity.max(0.0)).sum::<f32>() / denom).clamp(0.0, 1.0);
    (matches, local_score)
}

fn best_match(token: &[f32], others: &[&[f32]]) -> (usize, f32) {
    others
        .iter()
        .enumerate()
        .map(|(i, other)| (i, common::cosine(token, other)))
        .fold(
            (0, f32::MIN),
            |best, cur| if cur.1 > best.1 { cur } else { best },
        )
}

/// Maps a patch index (row-major over `grid_h` x `grid_w`) to a pixel
/// region on a crop of `crop_w` x `crop_h`, assuming a uniform patch grid.
pub fn patch_region(index: usize, grid_h: u32, grid_w: u32, crop_w: u32, crop_h: u32) -> Region {
    if grid_w == 0 || grid_h == 0 {
        return Region {
            x: 0,
            y: 0,
            w: crop_w,
            h: crop_h,
        };
    }
    let row = (index as u32) / grid_w;
    let col = (index as u32) % grid_w;
    let cell_w = crop_w as f32 / grid_w as f32;
    let cell_h = crop_h as f32 / grid_h as f32;
    let x = (col as f32 * cell_w).round() as u32;
    let y = (row as f32 * cell_h).round() as u32;
    Region {
        x: x.min(crop_w.saturating_sub(1)),
        y: y.min(crop_h.saturating_sub(1)),
        w: (cell_w.round() as u32)
            .max(1)
            .min(crop_w - x.min(crop_w.saturating_sub(1))),
        h: (cell_h.round() as u32)
            .max(1)
            .min(crop_h - y.min(crop_h.saturating_sub(1))),
    }
}
