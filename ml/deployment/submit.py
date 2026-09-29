"""Jury entry point: CSVs + images -> submission.csv, embeddings.npy, candidates.csv.

Writes the pinned lct-offline driver config for the baked model and runs the native
extractor (decode/crop/resize/normalize/forward/L2 in Rust, ONNX Runtime CUDA).
Offline: nothing is downloaded at run time.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = Path("/opt/lct/model")


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads((MODEL / "model.json").read_text("utf-8"))
    graph, policy = MODEL / manifest["file"], MODEL / "policy.json"
    if sha(graph) != manifest["sha256"]:
        raise SystemExit(f"model checksum mismatch: {graph}")
    retrieval = manifest["retrieval"]
    if (retrieval["method"], retrieval["k"], retrieval["alpha"]) != ("gallery_dba", 4, 2):
        raise SystemExit("Native submission requires gallery_dba k=4 alpha=2")

    import onnxruntime
    runtime = Path(onnxruntime.__file__).parent / "capi/libonnxruntime.so.1.24.4"
    config = {
        "schema_version": 1,
        "query_csv": str(args.query.resolve()),
        "gallery_csv": str(args.gallery.resolve()),
        "images_dir": str(args.images.resolve()),
        "runtime": {"path": str(runtime), "sha256": sha(runtime)},
        "models": [{
            "graph": {"path": str(graph), "sha256": manifest["sha256"]},
            "preprocessing": {"size": manifest["input_height"], "mean": manifest["mean"],
                              "std": manifest["std"], "mode": "stretch", "crop_pct": 1.0},
            "metadata_policy": {"path": str(policy), "sha256": sha(policy)},
        }],
        "threshold": retrieval["threshold"] if args.threshold is None else args.threshold,
        "threshold_status": ("frozen_calibration" if args.threshold is None and
                             retrieval.get("threshold_status", "").startswith("frozen_calibration")
                             else "development_only"),
        "batch_size": args.batch_size,
        "preprocessing_options": {"workers": 4, "reuse_identical": True},
    }
    # lct-offline refuses to overwrite, so it writes a fresh directory next to --out.
    args.out.mkdir(parents=True, exist_ok=True)
    fresh = Path(tempfile.mkdtemp(dir=args.out)) / "run"
    config_path = fresh.parent / "config.json"
    config_path.write_text(json.dumps(config, indent=1), "utf-8")
    subprocess.run([sys.executable, "/opt/lct/launch-offline.py", str(config_path), str(fresh)],
                   check=True)
    for name in ("submission.csv", "candidates.csv", "embeddings.npy"):
        os.replace(fresh / name, args.out / name)
    shutil.rmtree(fresh.parent)


if __name__ == "__main__":
    main()
