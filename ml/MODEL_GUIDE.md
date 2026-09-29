# Model and inference guide

The shipped model is **vitl16_dinov3_veriwild_tt_2plus2_fitv8, 1.2.0-final**.
It is the exported checkpoint behind the historical 91.956359% SEARCH result.
The current native CUDA pipeline scores **92.004809% SEARCH mAP@10**;
the previously opened audit fold scores **87.003299%**. Neither is a jury score.
Full evidence, tradeoffs and limitations: [submission verification](../docs/submission-readiness.md).

## Representation and retrieval

Clamp the supplied bbox to the frame, decode RGB, crop and stretch to 256×256
using Pillow-compatible bicubic interpolation. Normalize channels with mean
[0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. DINOv3 ViT-L/16
uses its CLS token and a trained BN neck, then L2 normalization to 1,024 dimensions.
The graph mixes FP16 weights/activations with FP32 sensitive operations and residuals.
The training classifier is removed. Images and bboxes are the only model inputs.

Offline ranking uses static gallery-only DBA: for each gallery vector, select
itself plus four nearest gallery neighbors, weight positive cosine similarities
to power two, sum and normalize. Compare each independent raw query to that
transformed gallery. Stable ties follow gallery input order. No query expansion,
query clustering, cross-query state, OCR or camera/time metadata is used.
Raw embeddings, not DBA vectors, are written to `embeddings.npy`.

Always emit the top ten gallery IDs in `submission.csv`, even for refused queries.
Emit an accepted top-1 row in `candidates.csv` only when the score is >=
**0.720700740814209**. The live pgvector service uses raw cosine and threshold
**0.6959864497184753**, with version-specific galleries. These scores are not probabilities.
Thresholds maximize calibration Q=0.7*F1+0.3*TNR at a 20% unknown prior, then are frozen.

## Training lineage

1. Public DINOv3 ViT-L/16, pinned timm foundation revision and license.
2. Official VERI-Wild TRAIN continuation: two epochs, 34,726 updates, then one
   TRAIN+public-TEST continuation: 416,314 images / 40,671 IDs, 26,020 updates.
3. Fresh 928-class head; contest FIT-v8, 5,722 crops; camera-aware 2+2 sampler,
   eight epochs / 2,864 updates, fixed seed 20260920. Metric learning combines
   label-smoothed classification and batch-hard triplet loss.
4. CPU ONNX export and mixed-precision conversion; independent FP32 reference
   and actual Linux CUDA placement checks, then frozen calibration and audit.

The fresh FIT retrain reached 91.212704% SEARCH (-0.743655 percentage points from
the archived BF16 high). Full external training was not repeated. Historical
24 FIT exclusions and the limits of exact overlap detection are disclosed in
[TRAINING.md](TRAINING.md); no new pruning was introduced.

The graph SHA-256 is `a1a63bcbfc00388ad6612a33c4861ffae8684a67bb3c0f1cb0d380b0b4bec1e0`.
It occupies 607,325,498 bytes. [models/model.json](../models/model.json) is the
authoritative preprocessing/threshold manifest; `models/policy.json` binds
the permitted CPU shape operations to the exact Linux ORT binary and graph.

## Runtime and limitations

`just submit-image` builds the offline native jury image and accelerated Python
path. `just perf-gpu` verifies the full extraction boundary and repeated outputs.
Final local results: native 22.417 ms / 181.549 FPS, Python 24.99 ms / 119.0 FPS.
This is RTX 3090 evidence, not measured A5000/535 compatibility. CUDA is mandatory
inside the jury image; silent CPU fallback is prohibited and PTX JIT is disabled.

SEARCH was used repeatedly for development. The calibration/audit roles had
already been opened for the previous model. They remain separate from training
but do not support an untouched-test claim. Plate-zone masking reduced SEARCH
mAP by 0.7124 points; this approximate diagnostic is not the jury mask test.
