# Selected inference model

`model.onnx` is DINOv3 ViT-L/16 + trained BN neck, version **1.2.0-final**,
`vitl16_dinov3_veriwild_tt_2plus2_fitv8`, with a 256×256 RGB input and a
1,024-dimensional L2-normalized float32 output. Size: **607,325,498 bytes**.

SHA-256: `a1a63bcbfc00388ad6612a33c4861ffae8684a67bb3c0f1cb0d380b0b4bec1e0`.
Fetch the actual graph using `git lfs pull`; a Git LFS pointer cannot run inference.
No other checkpoint is needed at runtime. `policy.json` binds the reviewed
Linux CUDA placement proof to this graph and ONNX Runtime 1.24.4.

`model.json` stores two calibrated thresholds: gallery DBA **0.720700740814209**
for offline submission; raw cosine **0.6959864497184753** for the live service.
Neither is a probability. Both are frozen from the calibration episode.

Read [model guide](../ml/MODEL_GUIDE.md), [training](../ml/TRAINING.md) and
[final verification](../docs/submission-readiness.md) for provenance and evidence.
The DINOv3 license is in [DINOV3_LICENSE.md](DINOV3_LICENSE.md). Public VERI-Wild
TRAIN and TEST were used for training, as allowed by organizer Q46; no contest
test data was used for fitting. A live gallery must be re-embedded for this model
version; vectors from the previous model must not be mixed with the new graph.
