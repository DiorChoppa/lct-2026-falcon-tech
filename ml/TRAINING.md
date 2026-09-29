# Training and reproducibility

The selected TT lineage is DINOv3 → VERI-Wild TRAIN E2 → VERI-Wild TRAIN+public
TEST E1 → contest FIT-v8 eight epochs. The exported graph is version 1.2.0-final,
SHA-256 `a1a63bcbfc00388ad6612a33c4861ffae8684a67bb3c0f1cb0d380b0b4bec1e0`.
Public external test training is allowed by organizer Q46. **Contest jury test,
SEARCH, calibration and audit are never training inputs.**

## Inputs, licenses and exact lineage

| Artifact | SHA-256 / source |
|---|---|
| DINOv3 foundation | `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941` |
| VERI-Wild TRAIN E2 backbone | `82b46d0d642ecac828ed2e3e6482a680270bed2a374e1fc634969d1ee58ad641` |
| TRAIN+TEST E1 backbone | `4abc470e140efdd372ee0b791e537cdef401ba5b4cdb9f6fbfae301a487d69d4` |
| Terminal FIT weights-only checkpoint | `26032a04664494ddf70e4cabf51b5a87b4ffee1ec90c637b29c2d661af6de772` |
| Frozen FIT-v8 CSV | `fe1394a9fbc7ba5e21eeed802252d373348940ebece6d0ae5a6c4727215ff962` |

Foundation: [timm model at revision 30c1109](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/resolve/30c1109559f65dea34316b0d4842d35c5771fe11/model.safetensors),
1,212,347,640 bytes, under [DINOv3 License](../models/DINOV3_LICENSE.md).
External images and metadata: [official VERI-Wild release](https://github.com/PKU-IMRE/VERI-Wild/issues/16),
for noncommercial research/education. Obtain these from the authors; raw datasets
and intermediate training weights are not shipped inside the submission.
Needed metadata: `train_list_start0.txt`, `test_10000_id.txt`,
`test_10000_id_query.txt`, and `vehicle_info.txt`.

All 277,797 TRAIN images (30,671 IDs) and 138,517 public TEST images (10,000 IDs)
were freshly rehashed. Across the resulting 416,314 images, encoded-file/RGB/crop
overlap with all 11,416 contest images is zero. The portable TT loader's full
ordered rows/labels/cameras match the archived run with zero differences.
This is exact matching, not a perceptual or foundation-pretraining guarantee.

FIT-v8 has 5,722 rows and 928 identities. The historical pipeline excluded 24
of the 5,746 FIT-role rows before the later no-pruning policy. This reproduction
retains that frozen input and introduces no new exclusion or relabeling. The
provided FIT and episode manifests preserve the original split; the upstream
EDA process that originally constructed those roles is not fully reconstructible
from this smaller submission repository. `train/fit_v8.py` checks the frozen result.

## Environment

Training: Python 3.12, torch 2.14.0+cu126, torchvision 0.29.0+cu126, timm 1.0.29,
safetensors 0.8.0, CUDA-capable Ampere GPU (original RTX 3090). Example setup:

```sh
cd ml
uv sync --group dev
uv pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install timm==1.0.29 safetensors==0.8.0 onnx==1.23.0
```

Export uses the ONNX Runtime **1.24.4** float16 converter. Newer converters can
change graph node names and invalidate the reviewed placement policy. Runtime
dependencies are pinned separately in `requirements-submit.txt` and the Dockerfile.

## Reproduction commands

From the repository root, with organizer JPEGs in `dataset/images`:

```sh
just train-check
just train-veriwild /data/VERI-Wild /data/dinov3.safetensors
# Retain the terminal E2 full checkpoint: its classifier and AdamW state seed TT.
cd ml
python -m train.stage1_tt --metadata-dir /data/metadata   --train-images /data/train/images --test-images /data/test/images   --init-backbone runs/stage1_veriwild/backbone.safetensors   --init-head runs/stage1_veriwild/checkpoints/epoch0002-batch0000-step0034726.pt   --contest-images ../dataset/images --run-dir runs/stage1_tt
python -m train.stage2_fit --config train/configs/stage2_fit_v8_2plus2.json   --weights runs/stage1_tt/backbone.safetensors --run-dir runs/stage2_tt_fit
cd ..
just train-export runs/stage2_tt_fit/checkpoints/epoch0008-batch0000-step0002864.pt
```

The example external layout contains author-provided identity directories. E2
training needs the official TRAIN layout expected by `stage1_veriwild`; TT accepts
separate TRAIN/TEST image roots and metadata. `stage1_tt --dry-run` validates the
complete ordered corpus without launching GPU training. A real TT run also
rehashes contest and external images and rejects exact overlap before fitting.

TRAIN E2: B16=P4×K4, two epochs, 34,726 updates. Historical cumulative sampled
coverage is 144,710 images / 27,528 IDs after epoch 1 and 189,014 / 30,383 after
epoch 2. The repaired namespaced sampler reproduces those counts exactly.
TT E1: seed 20260923, one epoch / 26,020 updates, reuse 30,671 old classifier rows
and AdamW moments, initialize 10,000 new rows/moments, retain trained BN neck.
Historical TT duration was 115.306 minutes. FIT resets a fresh 928-class head:
eight epochs / 2,864 updates, seed 20260920, B16, camera-aware 2+2 sampler,
256-square stretch, BF16, AdamW backbone/head LR 3e-5/3e-4, weight decay .05,
50-step warmup then constant LR, label smoothing .1, triplet margin .3, clip 5,
two DataLoader workers. The inference classifier is removed during export.

## What was actually repeated

On 29 September the archived TT external parent was verified and the entire FIT
stage was rerun: 8 epochs, 2,864 updates, 809.124 training seconds, 76 C peak.
The new model achieved **91.212704% SEARCH mAP@10**, versus the original
**91.956359%** (-0.743655 points). The original checkpoint remains the shipped
model. First-step loss/gradient agreed exactly; later CUDA training diverged.
Full external/foundation training was not repeated, and bit-identical weights
are not promised. Portable external code was validated by full input/coverage
replay and optimizer-state tests; it is a documented reconstruction.

`reid_model.py`, `checkpoint_inference.py` and `residual.py` retain the original
training/export source. The portable trainer removes campaign-specific SEARCH
evaluation, cache and deadline plumbing; that evaluation happened after terminal
weights were saved. The fresh research-side FIT run used verified source/input
snapshots and a pixel-identical cache. See `validation/final_20260929/`.

Newly trained weights are a new candidate. Re-export, audit numerical/placement
parity, freeze selection, calibrate and evaluate on appropriate new held-out data.
Do not reuse the shipped thresholds or old untouched-audit claims for new weights.
The retained training checkpoint and external parents stay in the playground,
outside the jury-scanned weight directory, to respect the 2 GB inference cap.
