import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('native_calibration', SCRIPTS / 'calibrate_native.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def test_score_ties_are_indivisible():
    selected, _ = calibration.choose([.8, .8, .7], [True, False, True], [True, False, True])
    assert selected['threshold'] == .7
    assert (selected['TP'], selected['FP']) == (2, 1)


def test_known_and_unknown_separation_keeps_exact_native_boundary():
    selected, _ = calibration.choose([.9, .2], [True, False], [True, False])
    assert selected['threshold'] == .9
    assert selected['Q'] == 1


def test_reject_all_is_an_available_threshold():
    scores = np.array([.9, .8], dtype=np.float32)
    selected, _ = calibration.choose(scores, [True, False], [False, False])
    assert selected['threshold'] > float(scores.max())
    # Replay the manifest conversion used by the live Rust service.
    assert not (scores >= np.float32(selected['threshold'])).any()
    assert (selected['TP'], selected['FP'], selected['FN'], selected['TN']) == (0, 0, 1, 1)


def test_both_policies_use_raw_embeddings_without_candidate_csv(tmp_path):
    episode, output = tmp_path / 'episode', tmp_path / 'output'
    episode.mkdir()
    output.mkdir()
    query_ids = [f'q{i}' for i in range(200)]
    gallery_ids = [f'g{i}' for i in range(750)]
    query_vehicles = list(range(160)) + list(range(800, 840))
    pd.DataFrame({'image_id': query_ids}).to_csv(episode / 'query.csv', index=False)
    pd.DataFrame({'image_id': gallery_ids}).to_csv(episode / 'gallery.csv', index=False)
    pd.DataFrame({'image_id': query_ids + gallery_ids,
                  'vehicle_id': query_vehicles + list(range(750)),
                  'camera_id': [0] * 200 + [1] * 750,
                  'split': ['query'] * 200 + ['gallery'] * 750}).to_csv(
        episode / 'ground_truth.csv', index=False)
    basis = np.eye(1024, dtype=np.float32)
    matrix = np.concatenate([basis[query_vehicles], basis[:750]])
    np.save(output / 'embeddings.npy', matrix)
    report = calibration.calibrate(output, episode)
    for name in ('plain', 'dba'):
        assert report[name]['threshold'] == 1
        assert report[name]['Q'] == 1
        assert report[name]['TP'] == 160
        assert report[name]['TN'] == 40
    matrix[0] *= .9
    np.save(output / 'embeddings.npy', matrix)
    with pytest.raises(ValueError, match='unit vectors'):
        calibration.calibrate(output, episode)
