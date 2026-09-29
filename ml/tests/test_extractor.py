"""extractor.py против эталона паритета ml/parity/ (текущий models/model.onnx).

Нужны models/model.onnx и dataset/images — без них тесты пропускаются.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="Install requirements-submit.txt for runtime checks")

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "models" / "model.json"
PARITY = ROOT / "ml" / "parity"

needs_model = pytest.mark.skipif(
    not (ROOT / "models" / "model.onnx").exists() or not (ROOT / "dataset" / "images").exists(),
    reason="нет models/model.onnx или dataset/images",
)


@pytest.fixture(scope="module")
def extractor():
    from extractor import Extractor

    return Extractor(MANIFEST, providers=["CPUExecutionProvider"])


@needs_model
def test_extract_matches_parity_reference(extractor):
    src = json.loads((PARITY / "source.json").read_text())
    ref = np.load(PARITY / "embedding.npy")
    got = extractor.extract(ROOT / "dataset" / "images" / f"{src['image_id']}.jpg", src["bbox"])
    assert got.shape == (extractor.dim,)
    assert float(got @ ref) > 0.999
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-4)


@needs_model
def test_bbox_outside_frame_raises(extractor):
    src = json.loads((PARITY / "source.json").read_text())
    with pytest.raises(ValueError):
        extractor.extract(
            ROOT / "dataset" / "images" / f"{src['image_id']}.jpg", (5000, 5000, 10, 10)
        )


def test_jury_mode_rejects_missing_cuda_before_loading_weights(monkeypatch):
    import extractor as module

    monkeypatch.setenv("LCT_REQUIRE_CUDA", "1")
    monkeypatch.setattr(module.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module.Extractor(MANIFEST)


def test_requested_cuda_cannot_silently_initialize_cpu(monkeypatch):
    import extractor as module

    class CpuSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(module.ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(module.ort, "InferenceSession", lambda *a, **k: CpuSession())
    with pytest.raises(RuntimeError, match="initialized CPUExecutionProvider"):
        module.Extractor(MANIFEST, providers=["CUDAExecutionProvider"])
