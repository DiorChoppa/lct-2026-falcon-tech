"""extract(): эмбеддинг ТС по кадру и bbox — то, что замеряет жюри (07-organizers-qa.md §1).

Полный цикл внутри extract(): чтение файла, декодирование, кроп, препроцессинг, forward,
L2-нормализация. Модель и препроцессинг — из models/model.json (тот же ONNX, что у сервиса).

    from extractor import Extractor
    ex = Extractor()                                   # models/model.json, CUDA если есть
    vec = ex.extract("dataset/images/<id>.jpg", (x, y, w, h))      # np.ndarray (D,), float32
    vecs = ex.extract_batch([(path, bbox), ...])                    # (N, D)

Контракт организаторов (extractor.py) на 21.09 не опубликован; сигнатура выше — наше
допущение, адаптер под их контракт добавляется здесь же, когда он появится.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

from reid.preprocess import load_manifest, read_crop, to_tensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "model.json"

BBox = tuple[int, int, int, int]


class Extractor:
    def __init__(
        self,
        manifest: Path | str = DEFAULT_MANIFEST,
        providers: list[str] | None = None,
        decode_workers: int | None = None,
    ):
        self.manifest_path = Path(manifest)
        self.m = load_manifest(self.manifest_path)
        model_path = self.manifest_path.parent / self.m["file"]
        if providers is None:
            available = ort.get_available_providers()
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
            ]
        require_cuda = os.environ.get("LCT_REQUIRE_CUDA") == "1"
        if not providers or (require_cuda and providers[0] != "CUDAExecutionProvider"):
            raise RuntimeError("The jury image requires CUDA; no CUDA execution provider is available")
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            # CUDA/cuDNN из pip-пакетов nvidia-* (образ жюри) вне пути поиска загрузчика:
            # без этого CUDA EP молча не поднимается и сессия уходит на CPU.
            ort.preload_dlls()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        if providers[0] != self.provider:
            raise RuntimeError(
                f"Requested {providers[0]}, but ONNX Runtime initialized {self.provider}; "
                "check the NVIDIA driver and container CUDA libraries"
            )
        self.session.disable_fallback()
        # Декод JPEG 1080p дороже forward на GPU: батч декодируется в потоках (PIL отпускает GIL)
        self.pool = ThreadPoolExecutor(decode_workers or min(32, os.cpu_count() or 4))

    @property
    def dim(self) -> int:
        return int(self.m["dim"])

    def load(self, path: Path | str, bbox: BBox) -> np.ndarray:
        """Чтение + декод + кроп + препроцессинг одного кадра → CHW float32."""
        return to_tensor(read_crop(path, *bbox), self.m)

    def forward(self, batch: np.ndarray) -> np.ndarray:
        out = self.session.run([self.m["output_name"]], {self.m["input_name"]: batch})[0]
        if not self.m.get("l2_normalized", False):
            out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out.astype(np.float32, copy=False)

    def extract(self, path: Path | str, bbox: BBox) -> np.ndarray:
        return self.forward(self.load(path, bbox)[None])[0]

    def extract_batch(self, items: list[tuple[Path | str, BBox]]) -> np.ndarray:
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        tensors = list(self.pool.map(lambda it: self.load(it[0], it[1]), items))
        return self.forward(np.stack(tensors))

    def warmup(self, n: int = 5) -> None:
        x = np.zeros((1, 3, self.m["input_height"], self.m["input_width"]), dtype=np.float32)
        for _ in range(n):
            self.forward(x)


_default: Extractor | None = None


def extract(path: Path | str, bbox: BBox) -> np.ndarray:
    """Функциональная форма с ленивой моделью по умолчанию — под возможный контракт жюри."""
    global _default
    if _default is None:
        _default = Extractor()
    return _default.extract(path, bbox)
