"""Reproduce v1.2 thresholds from fixed native calibration embeddings.

Use calibration labels only after model selection. Never tune on audit or jury
test data. Unlike the historical CPU midpoint report, v1.2 selects an attainable
native score boundary, maximizing Q, then TNR, then threshold.
"""
import argparse
import json
from pathlib import Path

import eval_holdout as evaluation
import numpy as np
from verify_submission import ids, native_gallery_dba, sequential_scores, sha256


def choose(scores, known, correct):
    scores = np.asarray(scores, dtype=np.float64)
    known, correct = np.asarray(known, dtype=bool), np.asarray(correct, dtype=bool)
    if scores.shape != known.shape or scores.shape != correct.shape or not np.isfinite(scores).all():
        raise ValueError('Matching finite score/outcome arrays required')
    if not known.any() or known.all():
        raise ValueError('Calibration requires both known and unknown queries')
    # The live service loads thresholds as f32. A float64-only increment would
    # round back to the maximum score and accept a query in the reject-all case.
    reject_all = float(np.nextafter(np.float32(scores.max()), np.float32(np.inf)))
    thresholds = np.r_[reject_all, np.unique(scores)]
    curve = [{'threshold': float(t), **evaluation.q_score(scores >= t, known, correct)} for t in thresholds]
    selected = max(curve, key=lambda r: (r['Q'], r['TNR'], r['threshold']))
    return selected, curve


def calibrate(submission, episode):
    query, gallery = evaluation.ev.load_gt(episode / 'ground_truth.csv')
    if query.index.tolist() != ids(episode / 'query.csv') or gallery.index.tolist() != ids(episode / 'gallery.csv'):
        raise ValueError('Ground-truth row order differs from the extraction input')
    known = np.array([bool(evaluation.ev.valid_positives(row, gallery)) for _, row in query.iterrows()])
    if (len(query), len(gallery), int(known.sum())) != (200, 750, 160):
        raise ValueError('Expected the fixed calibration episode with 20% unknown queries')
    matrix = np.load(submission / 'embeddings.npy', allow_pickle=False)
    if matrix.shape != (950, 1024) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError('Invalid native embeddings')
    if np.max(abs(np.linalg.norm(matrix.astype(np.float64), axis=1) - 1)) > 1e-5:
        raise ValueError('Native embeddings must be unit vectors')
    # Both policies derive scores from the same vectors and the independent
    # native scorer. CSV scores may be stale, edited, or already thresholded.
    report = {'curves': {}}
    for name, vectors in (('plain', matrix[200:]), ('dba', native_gallery_dba(matrix[200:]))):
        similarity = np.stack([sequential_scores(vectors, vector) for vector in matrix[:200]])
        positions = np.argmax(similarity, axis=1)
        correct = gallery.vehicle_id.to_numpy()[positions] == query.vehicle_id.to_numpy()
        report[name], report['curves'][name] = choose(similarity[np.arange(200), positions], known, correct)
    report['input_sha256'] = {name: sha256(episode / name) for name in ('query.csv', 'gallery.csv', 'ground_truth.csv')}
    report['embedding_sha256'] = sha256(submission / 'embeddings.npy')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(args.submission, args.episode)
    args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: report[name] for name in ('dba', 'plain')}, indent=2))


if __name__ == '__main__':
    main()
