"""LEGACY CPU/midpoint report for the previous model; use calibrate_native.py for v1.2.0.

Отложенная оценка замороженной модели: калибровка порога и аудит, косинус против DBA.

Эпизоды search / calibration / audit — `validation/<episode>/` (идентичности не пересекаются
с обучением и между собой, 20 % запросов без пары). Эмбеддинги — `embeddings.npy`
из `reid.submit` (строки query, затем gallery, как у жюри). Метрики считает немодифицированный
`organizers/evaluate.py`.

Порог выбирается ТОЛЬКО на calibration (максимум 0.7·F1 + 0.3·TNR, середина плато)
и один раз применяется к audit. Доверительные интервалы — бутстреп по идентичностям запросов.

  uv run python scripts/eval_holdout.py --runs <dir с search/ calibration/ audit/> \
      --json validation/holdout.json
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

from reid.artifacts import TOP_K, gallery_dba, l2

spec = importlib.util.spec_from_file_location("evaluate", ML / "organizers" / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

EPISODES = ("search", "calibration", "audit")
# пороги до калибровки: максимум Q на SEARCH (для сравнения в отчёте)
SEARCH_SELECTED = {"cosine": 0.6606993079185486, "dba": 0.6775355935096741}


def load(runs: Path, episode: str):
    root = ML / "validation" / episode
    q = pd.read_csv(root / "query.csv", dtype={"image_id": str})
    g = pd.read_csv(root / "gallery.csv", dtype={"image_id": str})
    query, gallery = ev.load_gt(root / "ground_truth.csv")
    emb = np.load(runs / episode / "embeddings.npy")
    if len(emb) != len(q) + len(g):
        raise ValueError(f"{episode}: {len(emb)} строк эмбеддингов на {len(q)}+{len(g)} кропов")
    return (
        q.image_id.tolist(),
        g.image_id.tolist(),
        l2(emb[: len(q)]),
        l2(emb[len(q) :]),
        query,
        gallery,
    )


def rank(q_emb, g_emb, g_ids, use_dba):
    sim = q_emb @ (gallery_dba(g_emb) if use_dba else g_emb).T
    order = np.argsort(-sim, axis=1, kind="stable")[:, :TOP_K]
    top1 = sim[np.arange(len(sim)), order[:, 0]]
    return [[g_ids[j] for j in row] for row in order], top1


def outcomes(q_ids, ranked, query, gallery):
    """По запросу: есть ли пара (после junk-фильтра) и верен ли верхний кандидат."""
    gal_vid = gallery.vehicle_id.to_dict()
    has, correct = [], []
    for qid, row in zip(q_ids, ranked):
        r = query.loc[qid]
        has.append(ev.valid_positives(r, gallery) > 0)
        correct.append(gal_vid[row[0]] == r.vehicle_id)
    return np.array(has), np.array(correct)


def q_score(accept, has, correct):
    tp = int((accept & has & correct).sum())
    fp = int((accept & ~(has & correct)).sum())
    fn = int((~accept & has).sum())
    tn = int((~accept & ~has).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    tnr = tn / (~has).sum() if (~has).any() else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "F1": f1, "TNR": tnr, "Q": 0.7 * f1 + 0.3 * tnr}


def pick_threshold(top1, has, correct):
    """Максимум Q по всем порогам между соседними top-1; середина плато максимума."""
    s = np.sort(np.unique(top1))
    cuts = np.concatenate([[s[0] - 1e-3], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-3]])
    qs = np.array([q_score(top1 >= t, has, correct)["Q"] for t in cuts])
    best = np.flatnonzero(np.isclose(qs, qs.max()))
    # самое длинное непрерывное плато среди максимумов, его середина
    runs, start = [], best[0]
    for a, b in itertools.pairwise(best):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, best[-1]))
    lo, hi = max(runs, key=lambda r: cuts[r[1]] - cuts[r[0]])
    return float((cuts[lo] + cuts[hi]) / 2), float(qs.max())


def evaluate(q_ids, ranked, top1, threshold, query, gallery):
    sub = dict(zip(q_ids, ranked))
    cand = {qid: [(row[0], float(s))] for qid, row, s in zip(q_ids, ranked, top1) if s >= threshold}
    m = ev.ranking_metrics(query, gallery, sub)
    c = ev.candidate_metrics(query, gallery, cand)
    return {
        "mAP@10": m["mAP@10"],
        "Rank-1": m["Rank-1"],
        "Rank-5": m["Rank-5"],
        **{k: c[k] for k in ("TP", "FP", "FN", "TN", "F1", "TNR", "PR-AUC")},
        "Q": 0.7 * c["F1"] + 0.3 * c["TNR"],
    }


def bootstrap(q_ids, ranked, top1, threshold, query, gallery, reps=2000, seed=20260927):
    """95 % интервалы mAP@10 и Q; единица пересэмплирования — идентичность запроса."""
    rng = np.random.default_rng(seed)
    gal_vid = gallery.vehicle_id.to_dict()
    gal_cam = gallery.camera_id.to_dict()
    ap, known = {}, {}
    for qid, row in zip(q_ids, ranked):
        r = query.loc[qid]
        n_pos = ev.valid_positives(r, gallery)
        known[qid] = n_pos > 0
        if n_pos:
            clean = ev.strip_junk(row, r, gal_vid, gal_cam)[:TOP_K]
            rel = np.array([gal_vid[x] == r.vehicle_id for x in clean], bool)
            prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
            ap[qid] = float((prec * rel).sum() / min(n_pos, TOP_K))
    has, correct = outcomes(q_ids, ranked, query, gallery)
    accept = top1 >= threshold
    ids = np.array([query.loc[q].vehicle_id for q in q_ids])
    groups = {v: np.flatnonzero(ids == v) for v in np.unique(ids)}
    keys = list(groups)
    maps, qs = [], []
    for _ in range(reps):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        aps = [ap[q_ids[i]] for i in idx if known[q_ids[i]]]
        maps.append(np.mean(aps))
        qs.append(q_score(accept[idx], has[idx], correct[idx])["Q"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"mAP@10_ci95": ci(maps), "Q_ci95": ci(qs), "reps": reps, "unit": "query identity"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ML.parent / "models" / "model.json")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    m = json.loads(args.manifest.read_text())
    manifest = {"cosine": float(m["threshold"]), "dba": float(m["retrieval"]["threshold"])}

    data = {e: load(args.runs, e) for e in EPISODES}
    report = {
        "model": m["name"],
        "version": m["version"],
        "manifest_threshold": manifest,
        "search_selected_threshold": SEARCH_SELECTED,
        "methods": {},
    }
    for method, use_dba in (("cosine", False), ("dba", True)):
        res = {}
        ranked = {}
        for e, (q_ids, g_ids, qe, ge, query, gallery) in data.items():
            ranked[e] = rank(qe, ge, g_ids, use_dba)
        q_ids, _, _, _, query, gallery = data["calibration"]
        has, correct = outcomes(q_ids, ranked["calibration"][0], query, gallery)
        t_cal, q_cal = pick_threshold(ranked["calibration"][1], has, correct)
        res["threshold_calibrated"] = t_cal
        res["calibration_Q_at_calibrated"] = q_cal
        for e, (q_ids, _, _, _, query, gallery) in data.items():
            rk, top1 = ranked[e]
            res[e] = {
                "search_selected": evaluate(
                    q_ids, rk, top1, SEARCH_SELECTED[method], query, gallery
                ),
                "calibrated": evaluate(q_ids, rk, top1, t_cal, query, gallery),
                "manifest": evaluate(q_ids, rk, top1, manifest[method], query, gallery),
            }
            has, correct = outcomes(q_ids, rk, query, gallery)
            res[e]["queries"] = [  # для графиков порога в вебе: уверенность топ-1 и исход
                {"c": round(float(s), 4), "g": "absent" if not h else "correct" if ok else "wrong"}
                for s, h, ok in sorted(zip(top1, has, correct), key=lambda t: t[0])
            ]
        q_ids, _, _, _, query, gallery = data["audit"]
        rk, top1 = ranked["audit"]
        res["audit"]["bootstrap_search_selected"] = bootstrap(
            q_ids, rk, top1, SEARCH_SELECTED[method], query, gallery
        )
        res["audit"]["bootstrap_calibrated"] = bootstrap(q_ids, rk, top1, t_cal, query, gallery)
        report["methods"][method] = res

    fmt = "{:<8} {:<12} {:<10} {:>7} {:>7} {:>7} {:>7} {:>7}  {}"
    print(fmt.format("метод", "эпизод", "порог", "mAP@10", "R1", "F1", "TNR", "Q", "TP/FP/FN/TN"))
    for method, res in report["methods"].items():
        for e in EPISODES:
            for kind in ("search_selected", "calibrated"):
                r = res[e][kind]
                print(
                    fmt.format(
                        method,
                        e,
                        kind,
                        *(f"{r[k]:.4f}" for k in ("mAP@10", "Rank-1", "F1", "TNR", "Q")),
                        f"{r['TP']}/{r['FP']}/{r['FN']}/{r['TN']}",
                    )
                )
        print(
            f"{method}: порог по calibration {res['threshold_calibrated']:.4f} "
            f"(в манифесте {manifest[method]:.4f}); audit mAP@10 95% ДИ "
            f"{res['audit']['bootstrap_calibrated']['mAP@10_ci95']}"
        )
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
