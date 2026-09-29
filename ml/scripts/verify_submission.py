"""Verify all three jury files against their inputs and the frozen native scorer.

No labels, camera data, or model fitting. Uses the native scorer's sequential
float32 dot products and float64 DBA accumulation to check the actual artifacts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        result = [row["image_id"] for row in csv.DictReader(stream)]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"Empty or duplicate image IDs: {path}")
    return result


def sequential_scores(gallery, query):
    return np.cumsum(gallery * query, axis=1, dtype=np.float32)[:, -1]


def native_gallery_dba(gallery):
    output = np.empty_like(gallery)
    for i, row in enumerate(gallery):
        scores = sequential_scores(gallery, row)
        neighbors = np.argsort(-scores, kind="stable")[:5]
        pooled = np.zeros(gallery.shape[1], dtype=np.float64)
        for j in neighbors:
            pooled += float(max(scores[j], 0)) ** 2 * gallery[j].astype(np.float64)
        norm = np.sqrt(np.cumsum(pooled * pooled, dtype=np.float64)[-1])
        output[i] = pooled / norm
    return output


def verify(directory, query_csv, gallery_csv, manifest):
    directory = Path(directory)
    qids, gids = ids(query_csv), ids(gallery_csv)
    if set(qids) & set(gids):
        raise ValueError("Query and gallery image IDs overlap")
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    policy = m["retrieval"]
    if (policy["method"], policy["k"], policy["alpha"]) != ("gallery_dba", 4, 2):
        raise ValueError("Unsupported frozen retrieval policy")
    threshold = float(policy["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite threshold")
    matrix = np.load(directory / "embeddings.npy", allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (len(qids) + len(gids), m["dim"]):
        raise ValueError(f"Invalid embeddings: {matrix.shape}, {matrix.dtype}")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.isfinite(matrix).all() or np.max(abs(norms - 1)) > 1e-5:
        raise ValueError("Embeddings must be finite unit vectors")
    with (directory / "submission.csv").open(newline="") as stream:
        submission = list(csv.reader(stream))
    if len(submission) != len(qids):
        raise ValueError("submission.csv must contain exactly one row per query, without a header")
    with (directory / "candidates.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", "gallery_id", "confidence"]:
            raise ValueError("Incorrect candidates.csv header")
        candidates = list(reader)
    accepted = {row["query_id"]: row for row in candidates}
    if len(accepted) != len(candidates) or not set(accepted) <= set(qids):
        raise ValueError("Duplicate or unknown candidate query ID")
    gallery = native_gallery_dba(matrix[len(qids):])
    expected_accepted = 0
    maximum_confidence_error = 0.0
    for qid, embedding, row in zip(qids, matrix[:len(qids)], submission, strict=True):
        scores = sequential_scores(gallery, embedding)
        order = np.argsort(-scores, kind="stable")[:10]
        expected = [qid, *[gids[j] for j in order]]
        if row != expected:
            raise ValueError(f"Ranking/order mismatch for {qid}")
        score = float(scores[order[0]])
        accept = score >= threshold
        expected_accepted += accept
        if accept != (qid in accepted):
            raise ValueError(f"Refusal threshold mismatch for {qid}")
        if accept:
            candidate = accepted[qid]
            value = float(candidate["confidence"])
            error = abs(value - score)
            maximum_confidence_error = max(maximum_confidence_error, error)
            if candidate["gallery_id"] != expected[1] or not np.isfinite(value) or error > 1e-6:
                raise ValueError(f"Candidate/confidence mismatch for {qid}")
    return {"status": "passed", "query_rows": len(qids), "gallery_rows": len(gids),
            "embedding_shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "maximum_norm_error": float(np.max(abs(norms - 1))),
            "all_rankings_match_native_reference": True, "accepted_queries": expected_accepted,
            "refused_queries": len(qids) - expected_accepted, "threshold": threshold,
            "maximum_confidence_error": maximum_confidence_error,
            "sha256": {name: sha256(directory / name) for name in
                       ("submission.csv", "candidates.csv", "embeddings.npy")},
            "input_sha256": {"query": sha256(query_csv), "gallery": sha256(gallery_csv),
                             "manifest": sha256(manifest)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/model.json"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = verify(args.submission, args.query, args.gallery, args.manifest)
    text = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
