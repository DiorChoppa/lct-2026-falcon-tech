"""Organizer artifacts: independent-query rankings, accepted top-1 and raw embeddings.

Submission rankings optionally use static-gallery DBA. embeddings.npy always keeps
the original normalized query then gallery vectors, without gallery refinement.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = 10


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def gallery_dba(gallery: np.ndarray) -> np.ndarray:
    """Fixed static-gallery DBA: top five including self, positive cosine squared.

    Queries never enter this computation. Ties preserve original gallery order.
    Accumulation uses float64, matching the qualified native retrieval helper.
    """
    gallery = np.asarray(gallery, dtype=np.float32)
    if gallery.ndim != 2 or not all(gallery.shape) or not np.isfinite(gallery).all():
        raise ValueError("gallery must be a nonempty finite matrix")
    if not np.allclose(np.linalg.norm(gallery, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("gallery vectors must have unit norm")
    # Keep scratch memory linear in gallery size; the gallery is refined only once.
    result = np.empty_like(gallery)
    for i, vector in enumerate(gallery):
        scores = gallery @ vector
        indices = np.argsort(-scores, kind="stable")[:5]
        weights = np.maximum(scores[indices].astype(np.float64), 0) ** 2
        pooled = (weights[:, None] * gallery[indices].astype(np.float64)).sum(axis=0)
        result[i] = pooled / np.linalg.norm(pooled)
    return result


def submission_retrieval(manifest: dict, override: float | None = None) -> tuple[bool, float]:
    """Read the submission-specific policy, without changing live-service defaults."""
    retrieval = manifest.get("retrieval", {})
    method = retrieval.get("method", "cosine")
    if method not in ("cosine", "gallery_dba"):
        raise ValueError(f"unsupported retrieval method: {method}")
    use_dba = method == "gallery_dba"
    if use_dba and (retrieval.get("k") != 4 or retrieval.get("alpha") != 2):
        raise ValueError("qualified gallery DBA requires k=4 and alpha=2")
    threshold = override if override is not None else (
        retrieval["threshold"] if use_dba else manifest["threshold"]
    )
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return use_dba, threshold


def write_artifacts(
    out_dir: Path,
    q_ids: list[str],
    g_ids: list[str],
    q_emb: np.ndarray,
    g_emb: np.ndarray,
    threshold: float,
    top_k: int = TOP_K,
    *,
    use_dba: bool = False,
) -> dict[str, Path]:
    if not g_ids or top_k < 1:
        raise ValueError("a nonempty gallery and positive top_k are required")
    if len(q_emb) != len(q_ids) or len(g_emb) != len(g_ids):
        raise ValueError("embedding rows must match image IDs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_emb, g_emb = l2(q_emb), l2(g_emb)
    sim = q_emb @ (gallery_dba(g_emb) if use_dba else g_emb).T
    order = np.argsort(-sim, axis=1, kind="stable")[:, :top_k]

    paths = {
        "embeddings": out_dir / "embeddings.npy",
        "submission": out_dir / "submission.csv",
        "candidates": out_dir / "candidates.csv",
    }
    np.save(paths["embeddings"], np.concatenate([q_emb, g_emb]).astype(np.float32))
    with paths["submission"].open("w", newline="") as f:
        w = csv.writer(f)
        for qid, row in zip(q_ids, order):
            w.writerow([qid, *(g_ids[j] for j in row)])
    with paths["candidates"].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, (qid, row) in enumerate(zip(q_ids, order)):
            score = float(sim[i, row[0]])
            if score >= threshold:
                w.writerow([qid, g_ids[row[0]], f"{score:.6f}"])
    return paths


def ground_truth(split: pd.DataFrame) -> pd.DataFrame:
    """GT в формате evaluate.py организаторов из CSV сплита (reid.split)."""
    gt = split[split["split"].isin(["query", "gallery"])]
    return gt[["image_id", "vehicle_id", "camera_id", "split"]].reset_index(drop=True)
