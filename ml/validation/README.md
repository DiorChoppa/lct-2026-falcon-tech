# Validation evidence

Authoritative selected-model reports: `metrics.json` (native SEARCH at the final
threshold), `holdout.json` (separate calibration/audit, raw cosine and DBA),
`container-verification.json`, and `plate_control.json` (approximate SEARCH masks).
Model 1.2.0-final: SEARCH **92.004809%**, calibration **82.999921%**, audit
**87.003299%** mAP@10. Audit Q **0.936485**, F1 **0.930693**, TNR **0.95**.

SEARCH was used adaptively. Calibration and audit were previously opened for
the incumbent; the selected model was frozen before its current calibration and
thresholds before its current audit. No untouched-holdout claim is made. The
identity-bootstrap interval in `holdout.json` keeps known/unknown mass at 160:40.

`final_20260929/summary.json` binds evidence to the graph, manifest, policy and
final Linux image. The directory contains raw timing samples, input integrity,
fresh FIT reproduction, native placement/numeric parity, query independence,
threshold freeze and independent format/ranking checks. `incumbent-*` and
`windows-*` files are historical contrary/diagnostic evidence, not current scores.

To reproduce final native threshold selection, use the frozen calibration
`embeddings.npy` (candidate CSV output is not needed):

```sh
cd ml
python scripts/calibrate_native.py --submission /data/calibration-output   --episode validation/calibration --json /data/calibration-replay.json
```

`scripts/eval_holdout.py` is the **legacy CPU/midpoint** evaluator for the previous
model; its SEARCH thresholds and bootstrap are historical. It must not overwrite
the current native report. `scripts/calibrate_native.py` recomputes both DBA and plain
scores from unit embeddings using the independent native arithmetic reference; exact observed score
boundaries reproduce the shipped values. The unchanged organizer evaluator is
in `organizers/evaluate.py` (SHA-256 `655c71db8c2e4d2cd7680c40c768afacfdffff360401111c1a46df921551ffa3`).

The episode CSVs contain labels only for evaluation; they never enter model
fitting. Public jury predictions in `../../submission/` have no available labels
and are checked only for validity, determinism and source binding.
