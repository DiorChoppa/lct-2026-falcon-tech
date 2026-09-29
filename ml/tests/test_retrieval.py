"""Synthetic-only checks for the qualified submission retrieval policy."""

import csv

import numpy as np
import pytest

from reid.artifacts import gallery_dba, l2, submission_retrieval, write_artifacts


def test_dba_matches_weighted_pool_and_leaves_raw_embeddings_unchanged(tmp_path):
    g = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    original = g.copy()
    refined = gallery_dba(g)
    expected = np.array([1.512, 0.384])
    np.testing.assert_allclose(refined[0], expected / np.linalg.norm(expected), atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(refined, axis=1), 1, atol=1e-7)
    np.testing.assert_array_equal(g, original)
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    paths = write_artifacts(tmp_path, ["q0", "q1"], ["g0", "g1", "g2"], q, g, 0,
                            use_dba=True)
    np.testing.assert_array_equal(np.load(paths["embeddings"]), np.concatenate([q, g]))
    with paths["candidates"].open(newline="") as stream:
        candidates = list(csv.DictReader(stream))
    assert candidates[0]["gallery_id"] == "g0"
    assert float(candidates[0]["confidence"]) == pytest.approx(refined[0, 0], abs=1e-6)


def test_query_order_independence_and_stable_gallery_ties(tmp_path):
    g = np.tile([[1.0, 0.0]], (12, 1)).astype(np.float32)
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    ids = [f"g{i}" for i in range(12)]
    a = write_artifacts(tmp_path / "a", ["q0", "q1"], ids, q, g, 1.01, use_dba=True)
    b = write_artifacts(tmp_path / "b", ["q1", "q0"], ids, q[::-1], g, 1.01, use_dba=True)
    assert a["submission"].read_text().splitlines() == b["submission"].read_text().splitlines()[::-1]
    assert a["submission"].read_text().splitlines()[0].split(",")[1:] == ids[:10]
    assert a["candidates"].read_text().splitlines() == ["query_id,gallery_id,confidence"]


def test_default_artifacts_remain_raw_cosine(tmp_path):
    rng = np.random.default_rng(2026)
    q, g = l2(rng.normal(size=(2, 8))), l2(rng.normal(size=(7, 8)))
    ids = [f"g{i}" for i in range(7)]
    paths = write_artifacts(tmp_path, ["q0", "q1"], ids, q, g, -1)
    expected = np.argsort(-(q @ g.T), axis=1, kind="stable")
    rows = list(csv.reader(paths["submission"].read_text().splitlines()))
    assert [r[1:] for r in rows] == [[ids[i] for i in row] for row in expected]


def test_policy_uses_dba_threshold_and_honors_override():
    manifest = {"threshold": 0.4, "retrieval": {
        "method": "gallery_dba", "k": 4, "alpha": 2, "threshold": 0.7,
    }}
    assert submission_retrieval(manifest) == (True, 0.7)
    assert submission_retrieval(manifest, 0.8) == (True, 0.8)
    assert submission_retrieval({"threshold": 0.4}) == (False, 0.4)
    with pytest.raises(ValueError, match="finite"):
        submission_retrieval(manifest, float("nan"))
    manifest["retrieval"]["k"] = 3
    with pytest.raises(ValueError, match="k=4"):
        submission_retrieval(manifest)


def test_empty_gallery_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        gallery_dba(np.zeros((0, 2), np.float32))
    with pytest.raises(ValueError, match="nonempty"):
        write_artifacts(tmp_path, ["q"], [], np.array([[1, 0]]), np.zeros((0, 2)), 0)
