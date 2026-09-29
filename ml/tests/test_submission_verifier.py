"""Artifact verification catches a missing refusal and preserves gallery-order ties."""
import csv
import json

import numpy as np
import pytest

from scripts.verify_submission import verify


def test_three_file_verifier_checks_ties_and_refusal(tmp_path):
    qids, gids = ["known", "unknown"], [f"g{i}" for i in range(10)]
    for name, identifiers in [("query", qids), ("gallery", gids)]:
        with (tmp_path / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows([identifier, 0, 0, 10, 10] for identifier in identifiers)
    manifest = {"dim": 2, "retrieval": {"method": "gallery_dba", "k": 4,
                                        "alpha": 2, "threshold": 0.65}}
    (tmp_path / "model.json").write_text(json.dumps(manifest))
    output = tmp_path / "submission"
    output.mkdir()
    np.save(output / "embeddings.npy", np.array([[1, 0], [0, 1], *[[1, 0]] * 10], np.float32))
    with (output / "submission.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([[qid, *gids] for qid in qids])
    candidates = output / "candidates.csv"
    candidates.write_text("query_id,gallery_id,confidence\nknown,g0,1.0\n")
    args = (output, tmp_path / "query.csv", tmp_path / "gallery.csv", tmp_path / "model.json")
    report = verify(*args)
    assert (report["accepted_queries"], report["refused_queries"]) == (1, 1)
    candidates.write_text(candidates.read_text() + "unknown,g0,0.0\n")
    with pytest.raises(ValueError, match="Refusal threshold mismatch"):
        verify(*args)
