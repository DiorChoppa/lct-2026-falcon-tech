# Independent Python / Rust embedding fixture

`crop.png` is a lossless FIT-only crop described by `source.json`.
`embedding.npy` is the selected checkpoint's independent **FP32 PyTorch, TF32-off**
reference, generated before calibration, not an ONNX self-reference. It is
float32 [1024], L2-normalized. The selected mixed ONNX graph must agree at cosine
>=0.999 in Python and the service Rust integration test. The source file pins
both the original checkpoint and the current ONNX hash. Batch precision can
produce tiny numerical differences; equality of embeddings is not required.
