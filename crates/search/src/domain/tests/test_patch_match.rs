use crate::domain::{match_patches, patch_region, Region};

#[test]
fn match_patches_pairs_mutual_nearest_neighbors() {
    let query = [1.0, 0.0, 0.0, 1.0]; // two tokens, dim 2
    let candidate = [1.0, 0.0, 0.0, 1.0];

    let (matches, local_score) = match_patches(&query, &candidate, 2);

    assert_eq!(2, matches.len());
    assert!((local_score - 1.0).abs() < 1e-6);
}

#[test]
fn match_patches_excludes_non_mutual_pairs() {
    // Two query tokens both prefer the single candidate token, but the
    // candidate's own best match is query token 0 — token 1 has no partner.
    let query = [1.0, 0.0, 0.9, 0.1];
    let candidate = [1.0, 0.0];

    let (matches, local_score) = match_patches(&query, &candidate, 2);

    assert_eq!(1, matches.len());
    assert_eq!(0, matches[0].query_index);
    assert_eq!(0, matches[0].candidate_index);
    assert!((local_score - 1.0).abs() < 1e-6);
}

#[test]
fn match_patches_returns_empty_for_empty_input() {
    let (matches, local_score) = match_patches(&[], &[1.0, 0.0], 2);

    assert!(matches.is_empty());
    assert_eq!(0.0, local_score);
}

#[test]
fn patch_region_covers_a_uniform_grid() {
    assert_eq!(
        Region {
            x: 0,
            y: 0,
            w: 2,
            h: 2
        },
        patch_region(0, 2, 2, 4, 4)
    );
    assert_eq!(
        Region {
            x: 2,
            y: 0,
            w: 2,
            h: 2
        },
        patch_region(1, 2, 2, 4, 4)
    );
    assert_eq!(
        Region {
            x: 0,
            y: 2,
            w: 2,
            h: 2
        },
        patch_region(2, 2, 2, 4, 4)
    );
    assert_eq!(
        Region {
            x: 2,
            y: 2,
            w: 2,
            h: 2
        },
        patch_region(3, 2, 2, 4, 4)
    );
}

#[test]
fn patch_region_stays_within_crop_bounds_for_non_divisible_grid() {
    let region = patch_region(8, 3, 3, 10, 10);
    assert!(region.x < 10 && region.y < 10);
    assert!(region.x + region.w <= 10);
    assert!(region.y + region.h <= 10);
}
