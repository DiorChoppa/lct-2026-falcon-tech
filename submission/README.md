# Public jury input predictions

Generated twice with the final offline Linux CUDA image; all three files are
byte-identical across runs. Model 1.2.0-final, graph SHA-256
`a1a63bcbfc00388ad6612a33c4861ffae8684a67bb3c0f1cb0d380b0b4bec1e0`.
Gallery DBA k=4 / alpha=2, threshold **0.720700740814209**.

- `submission.csv`: 1,110 rows, no header, each query ID followed by ten gallery IDs.
- `candidates.csv`: header `query_id,gallery_id,confidence`, accepted top-1 rows only.
- `embeddings.npy`: float32 [1860,1024], 1,110 query rows then 750 gallery rows,
  raw L2-normalized vectors in the organizer CSV order.

`run.json` records image/graph/manifest/policy and output hashes. The independent
verifier passed every ranking and refusal decision. No public-test quality score
is claimed because labels are unavailable. Build/run instructions are in the
[root README](../README.md); final evidence is in
[submission-readiness.md](../docs/submission-readiness.md).
