"""Метрики ReID по ТЗ §9 на валидационном split-е — те же определения, что в эталонном
`ml/organizers/evaluate.py` (сверено тестом tests/test_organizers_parity.py).

Ранжирование по запросам, у которых есть пара в галерее: mAP@10 (основная метрика жюри —
топ-10, AP нормирован на min(n_pos, 10)), mAP по полному ранжированию, Rank-1, Rank-5, mINP.
Кандидаты галереи той же машины с той же камеры, что запрос, исключаются до усечения
(кросс-камерный протокол жюри). Без camera_id ничего не исключается — так будет на тесте.

Режим отказа: решение на уровне запроса — принять топ-1 при сходстве ≥ порога или отказать.
TP — принят и топ-1 та же машина; FP — принят, а топ-1 чужой или пары нет; FN — пара есть,
но отказ; TN — пары нет и отказ. Precision = TP/(TP+FP), Recall = TP/(TP+FN), TNR = TN среди
запросов без пары. PR-AUC — average precision метки «у запроса есть пара» по скору принятого
top-1 (отказы в самом низу), как в candidates.csv при выбранном пороге.

Порядок эмбеддингов в .npy: сначала все строки split == "query" в порядке CSV, затем все
split == "gallery" — тот же порядок, что у embeddings.npy при сдаче.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = 10


def evaluate(
    sim: np.ndarray,
    q_ids: np.ndarray,
    g_ids: np.ndarray,
    q_cams: np.ndarray | None = None,
    g_cams: np.ndarray | None = None,
    threshold: float | None = None,
    min_tnr: float = 0.0,
    objective: str = "jury",
) -> dict:
    """sim — (Nq, Ng) косинусные сходства. threshold=None — выбрать порог с максимумом
    objective ("jury" = 0.7·F1 + 0.3·TNR, балл жюри за режим кандидатов; "f1") среди
    порогов с TNR ≥ min_tnr."""
    sim = np.asarray(sim, dtype=np.float64)
    q_ids, g_ids = np.asarray(q_ids), np.asarray(g_ids)
    n_q = sim.shape[0]
    same_id = q_ids[:, None] == g_ids[None, :]
    if q_cams is None or g_cams is None:
        valid = np.ones_like(same_id)
    else:
        valid = ~(same_id & (np.asarray(q_cams)[:, None] == np.asarray(g_cams)[None, :]))

    ap, ap10, inp, hit1, hit5 = [], [], [], [], []
    top1_score = np.full(n_q, -1.0)
    top1_correct = np.zeros(n_q, dtype=bool)
    has_match = np.zeros(n_q, dtype=bool)
    for i in range(n_q):
        # Сдаём сырое ранжирование (камер на тесте нет): top-1 для отказа и топ-10 берутся
        # из него, junk жюри выбрасывает уже из сданного списка.
        raw = np.argsort(-sim[i], kind="stable")
        top1_score[i], top1_correct[i] = sim[i][raw[0]], same_id[i][raw[0]]
        n_pos = int((same_id[i] & valid[i]).sum())
        if n_pos == 0:
            continue
        has_match[i] = True
        clean = raw[valid[i][raw]]
        ranks = np.flatnonzero(same_id[i][clean]) + 1
        ap.append(float(np.mean(np.arange(1, ranks.size + 1) / ranks)))
        inp.append(ranks.size / ranks[-1])
        top = raw[:TOP_K]
        rel = same_id[i][top[valid[i][top]]]
        ranks10 = np.flatnonzero(rel) + 1
        ap10.append(float(np.sum(np.arange(1, ranks10.size + 1) / ranks10) / min(n_pos, TOP_K)))
        hit1.append(bool(rel[:1].any()))
        hit5.append(bool(rel[:5].any()))

    curve = [
        _refusal_at(t, top1_score, top1_correct, has_match) for t in np.unique(top1_score)[::-1]
    ]
    if threshold is None:
        feasible = [p for p in curve if not p["tnr"] < min_tnr]
        if not feasible:
            best_tnr = max(p["tnr"] for p in curve)
            raise ValueError(f"ни один порог не даёт TNR ≥ {min_tnr}: максимум {best_tnr:.3f}")
        chosen = max(feasible, key=lambda p: (p[objective], p["threshold"]))
    else:
        chosen = _refusal_at(threshold, top1_score, top1_correct, has_match)

    return {
        "mAP10": float(np.mean(ap10)) if ap10 else float("nan"),
        "mAP": float(np.mean(ap)) if ap else float("nan"),
        "rank1": float(np.mean(hit1)) if hit1 else float("nan"),
        "rank5": float(np.mean(hit5)) if hit5 else float("nan"),
        "mINP": float(np.mean(inp)) if inp else float("nan"),
        "n_with_match": int(has_match.sum()),
        "n_without_match": int((~has_match).sum()),
        "n_gallery": int(sim.shape[1]),
        **chosen,
        "pr_auc": _pr_auc(top1_score, top1_score >= chosen["threshold"], has_match),
        "curve": curve,
        "top1_score": top1_score,
        "top1_correct": top1_correct,
        "has_match": has_match,
    }


def _refusal_at(t: float, score: np.ndarray, correct: np.ndarray, has_match: np.ndarray) -> dict:
    accept = score >= t
    # Верный по id, но junk (та же камера) при отсутствии других пар — это FP, как у жюри
    tp = int((accept & correct & has_match).sum())
    fn = int((~accept & has_match).sum())
    n_accept, n_neg = int(accept.sum()), int((~has_match).sum())
    precision = tp / n_accept if n_accept else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = float((~accept & ~has_match).sum() / n_neg) if n_neg else float("nan")
    return {
        "threshold": float(t),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tnr": tnr,
        # Балл жюри за режим кандидатов (07-organizers-qa.md §3); без open-set TNR = nan → 0
        "jury": 0.7 * f1 + 0.3 * (0.0 if np.isnan(tnr) else tnr),
    }


def _pr_auc(score: np.ndarray, accept: np.ndarray, has_match: np.ndarray) -> float:
    """Как pr_auc в evaluate.py организаторов: отказы получают скор ниже всех принятых."""
    n_pos = int(has_match.sum())
    if n_pos == 0 or not accept.any():
        return float("nan")
    s = np.where(accept, score, score[accept].min() - 1.0)
    hits = has_match[np.argsort(-s, kind="stable")]
    precision_at_k = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((precision_at_k * hits).sum() / n_pos)


def summary(result: dict) -> dict:
    """Только скаляры — для печати и metrics.json."""
    return {k: v for k, v in result.items() if isinstance(v, (int, float))}


def evaluate_split(split: pd.DataFrame, emb: np.ndarray, **kwargs) -> dict:
    """split — CSV от reid.split; emb — строки query, затем gallery, L2-нормируются здесь."""
    query = split[split["split"] == "query"]
    gallery = split[split["split"] == "gallery"]
    if len(emb) != len(query) + len(gallery):
        raise ValueError(
            f"в .npy {len(emb)} строк, а query + gallery = {len(query) + len(gallery)}"
        )
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    q_emb, g_emb = emb[: len(query)], emb[len(query) :]
    cams = "camera_id" in split.columns
    return evaluate(
        q_emb @ g_emb.T,
        query["vehicle_id"].to_numpy(),
        gallery["vehicle_id"].to_numpy(),
        query["camera_id"].to_numpy() if cams else None,
        gallery["camera_id"].to_numpy() if cams else None,
        **kwargs,
    )


def save_run(result: dict, out_dir: Path, config: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({**(config or {}), **summary(result)}, ensure_ascii=False, indent=2)
    )
    pd.DataFrame(result["curve"]).to_csv(out_dir / "threshold_curve.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", type=Path, required=True, help="CSV от reid.split")
    parser.add_argument("--emb", type=Path, required=True, help=".npy: query, затем gallery")
    parser.add_argument("--threshold", type=float, help="фиксированный порог отказа")
    parser.add_argument("--min-tnr", type=float, default=0.0, help="ограничение при выборе порога")
    parser.add_argument("--objective", choices=["jury", "f1"], default="jury")
    parser.add_argument("--out", type=Path, help="куда писать metrics.json и threshold_curve.csv")
    args = parser.parse_args()

    result = evaluate_split(
        pd.read_csv(args.split),
        np.load(args.emb),
        threshold=args.threshold,
        min_tnr=args.min_tnr,
        objective=args.objective,
    )
    print(json.dumps(summary(result), ensure_ascii=False, indent=2))
    if args.out:
        save_run(result, args.out)


if __name__ == "__main__":
    main()
