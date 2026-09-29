"""reid.eval и reid.artifacts против эталонного ml/organizers/evaluate.py на случайных данных.

Сплит с junk-парами (та же машина, та же камера), запросами без пары и порогом, который
часть запросов отклоняет; все числа обоих скриптов должны совпасть.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reid.artifacts import ground_truth, write_artifacts
from reid.eval import evaluate_split

ORGANIZERS = Path(__file__).resolve().parents[1] / "organizers" / "evaluate.py"


def _organizers():
    spec = importlib.util.spec_from_file_location("organizers_evaluate", ORGANIZERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(seed: int, n_q: int = 60, n_g: int = 40, dim: int = 8):
    rng = np.random.default_rng(seed)
    q = pd.DataFrame(
        {
            "image_id": [f"q{i}" for i in range(n_q)],
            "vehicle_id": rng.integers(0, 15, n_q),
            "camera_id": rng.integers(0, 3, n_q),
            "split": "query",
        }
    )
    g = pd.DataFrame(
        {
            "image_id": [f"g{i}" for i in range(n_g)],
            "vehicle_id": rng.integers(0, 12, n_g),
            "camera_id": rng.integers(0, 3, n_g),
            "split": "gallery",
        }
    )
    split = pd.concat([q, g], ignore_index=True)
    emb = rng.normal(size=(n_q + n_g, dim)).astype(np.float32)
    return split, emb


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_organizers_script(tmp_path, seed):
    split, emb = _synthetic(seed)
    q, g = split[split.split == "query"], split[split.split == "gallery"]
    ours = evaluate_split(split, emb)
    threshold = ours["threshold"]

    paths = write_artifacts(
        tmp_path, q.image_id.tolist(), g.image_id.tolist(), emb[: len(q)], emb[len(q) :], threshold
    )
    q.to_csv(tmp_path / "test_query.csv", index=False)
    g.to_csv(tmp_path / "test_gallery.csv", index=False)
    gt = ground_truth(split).set_index("image_id")
    gt_q, gt_g = gt[gt.split == "query"], gt[gt.split == "gallery"]

    org = _organizers()
    ranking = org.ranking_metrics(
        gt_q, gt_g, org.load_submission(paths["submission"], set(gt_g.index))
    )
    q_emb, g_emb, q_ids, g_ids = org.load_embeddings(
        paths["embeddings"], tmp_path / "test_query.csv", tmp_path / "test_gallery.csv"
    )
    full = org.full_ranking_metrics(q_emb, g_emb, q_ids, g_ids, gt_q, gt_g)
    cands = org.candidate_metrics(gt_q, gt_g, org.load_candidates(paths["candidates"]))

    assert ranking["n_scored"] == ours["n_with_match"]
    assert ranking["n_openset_excluded"] == ours["n_without_match"]
    assert ours["mAP10"] == pytest.approx(ranking["mAP@10"])
    assert ours["rank1"] == pytest.approx(ranking["Rank-1"])
    assert ours["rank5"] == pytest.approx(ranking["Rank-5"])
    assert ours["mAP"] == pytest.approx(full["mAP_full"], abs=1e-5)
    assert ours["mINP"] == pytest.approx(full["mINP"], abs=1e-5)
    assert ours["precision"] == pytest.approx(cands["Precision"])
    assert ours["recall"] == pytest.approx(cands["Recall"])
    assert ours["f1"] == pytest.approx(cands["F1"])
    assert ours["tnr"] == pytest.approx(cands["TNR"])
    assert ours["pr_auc"] == pytest.approx(cands["PR-AUC"], abs=1e-5)


def test_example_submission_has_expected_shape():
    # Пример организаторов: submission без заголовка, candidates с заголовком, отказ = нет строки
    example = ORGANIZERS.parent / "example_submission"
    with (example / "submission.csv").open() as f:
        first = f.readline().split(",")
    assert first[0].startswith("q_")
    cands = pd.read_csv(example / "candidates.csv")
    assert list(cands.columns) == ["query_id", "gallery_id", "confidence"]
    n_q = len(pd.read_csv(example / "test_query.csv"))
    assert cands.query_id.nunique() < n_q
