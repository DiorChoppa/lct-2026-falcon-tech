# Model, training and jury inference

The selected TT model scores 92.004809% SEARCH and 87.003299% on the previously
opened audit fold. Read [MODEL_GUIDE.md](MODEL_GUIDE.md), [TRAINING.md](TRAINING.md)
and [final verification](../docs/submission-readiness.md) for scope and limits.

```sh
git lfs pull
just submit-image
docker run --rm --network none --gpus all   -v /path/to/data:/data:ro -v "$PWD/submission:/out" reid-submit   --images /data/images --query /data/test_query.csv --gallery /data/test_gallery.csv --out /out
just perf-gpu
```

The native CUDA image checks the graph and placement policy and fails on missing
CUDA. The baked Python `extract(path, bbox)` path is also benchmarked. Output:
headerless `submission.csv`, accepted-only `candidates.csv`, raw float32
`embeddings.npy` in query-then-gallery order. Static gallery DBA k=4 / alpha=2 uses
threshold **0.720700740814209**; queries remain independent. The live pgvector
service uses plain cosine threshold **0.6959864497184753**.

`extractor.py` and `reid/` provide Python inference; `inference/` contains native
CUDA/offline Rust; `train/` contains source and export recipes; `validation/`
contains metrics and provenance. Run `pytest -q` and `ruff check .` inside `ml/`.
The separate offline crate has its own Cargo tests. `scripts/verify_submission.py`
independently verifies rankings, confidence thresholds, IDs and embedding norms.
