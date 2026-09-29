"""Сравнение двух наборов артефактов сдачи: Python-пайплайн (`reid.submit`) и Rust
(`reid-cli submit`). Паритет сервиса с артефактом жюри на всём тесте, а не на одном кропе.

  uv run python -m reid.compare ../submission ../submission-rust

Печатает: косинус между эмбеддингами одной строки (min / медиана), долю запросов с тем же
топ-1, с тем же топ-10 (как множество и как порядок), совпадение решений отказа.

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_submission(path: Path) -> dict[str, list[str]]:
    with path.open(newline="") as f:
        return {row[0]: row[1:] for row in csv.reader(f) if row}


def read_candidates(path: Path) -> dict[str, tuple[str, float]]:
    with path.open(newline="") as f:
        return {r["query_id"]: (r["gallery_id"], float(r["confidence"])) for r in csv.DictReader(f)}


def compare(a: Path, b: Path) -> dict:
    ea, eb = np.load(a / "embeddings.npy"), np.load(b / "embeddings.npy")
    if ea.shape != eb.shape:
        raise ValueError(f"embeddings.npy: {ea.shape} vs {eb.shape}")
    ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
    eb = eb / np.linalg.norm(eb, axis=1, keepdims=True)
    cos = (ea * eb).sum(axis=1)

    sa, sb = read_submission(a / "submission.csv"), read_submission(b / "submission.csv")
    if sa.keys() != sb.keys():
        raise ValueError("submission.csv: разные наборы query_id")
    q = list(sa)
    same_top1 = np.mean([sa[k][:1] == sb[k][:1] for k in q])
    same_top10_set = np.mean([set(sa[k]) == set(sb[k]) for k in q])
    same_top10_order = np.mean([sa[k] == sb[k] for k in q])

    ca, cb = read_candidates(a / "candidates.csv"), read_candidates(b / "candidates.csv")
    same_decision = np.mean([(k in ca) == (k in cb) for k in q])
    both = [k for k in q if k in ca and k in cb]
    same_candidate = np.mean([ca[k][0] == cb[k][0] for k in both]) if both else float("nan")
    conf_diff = max((abs(ca[k][1] - cb[k][1]) for k in both), default=0.0)

    return {
        "n_rows": int(ea.shape[0]),
        "n_query": len(q),
        "cos_min": float(cos.min()),
        "cos_median": float(np.median(cos)),
        "same_top1": float(same_top1),
        "same_top10_set": float(same_top10_set),
        "same_top10_order": float(same_top10_order),
        "accepted_a": len(ca),
        "accepted_b": len(cb),
        "same_decision": float(same_decision),
        "same_candidate_when_both_accepted": float(same_candidate),
        "max_confidence_diff": float(conf_diff),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a", type=Path, help="каталог с submission.csv, embeddings.npy, candidates.csv")
    p.add_argument("b", type=Path)
    p.add_argument("--min-cos", type=float, default=0.99, help="порог паритета по эмбеддингам")
    args = p.parse_args()
    r = compare(args.a, args.b)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r["cos_min"] < args.min_cos:
        raise SystemExit(f"паритет нарушен: min cos {r['cos_min']:.4f} < {args.min_cos}")


if __name__ == "__main__":
    main()
