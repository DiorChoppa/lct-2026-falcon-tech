"""reid.preprocess (read_crop + to_tensor) побитно против прежнего кода extractor.load().

Прежний путь: полный декод кадра Pillow, convert, crop, resize, (x/255 − mean)/std.
Реальные кадры нужны из dataset/images — без них тесты на кадрах пропускаются.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from reid.preprocess import RESAMPLE, crop_bbox, load_manifest, read_crop, to_tensor

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "dataset" / "images"
M = load_manifest(ROOT / "models" / "model.json")

needs_images = pytest.mark.skipif(not IMAGES.exists(), reason="нет dataset/images")


def _reference(path, bbox, m=M):
    """Код extractor.load() + to_tensor до оптимизации, дословно."""
    with Image.open(path) as img:
        img.draft("RGB", None)
        crop = crop_bbox(img.convert("RGB"), *bbox)
    size = (m["input_width"], m["input_height"])
    x = np.asarray(crop.convert("RGB").resize(size, RESAMPLE[m["resize"]]), dtype=np.float32)
    x = (x / 255.0 - np.asarray(m["mean"], dtype=np.float32)) / np.asarray(
        m["std"], dtype=np.float32
    )
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _assert_same(path, bbox):
    got, ref = to_tensor(read_crop(path, *bbox), M), _reference(path, bbox)
    assert got.dtype == ref.dtype == np.float32
    assert got.shape == ref.shape and got.flags.c_contiguous
    assert got.tobytes() == ref.tobytes(), (path, bbox)


def _test_rows(n: int = 40) -> list[tuple[Path, tuple[int, int, int, int]]]:
    df = pd.concat(
        [
            pd.read_csv(ROOT / "dataset" / f, dtype={"image_id": str})
            for f in ("test_query.csv", "test_gallery.csv")
        ]
    )
    bottom = df.y + df.h
    # равномерная выборка + края: bbox до низа кадра (полный декод) и самый верхний bbox
    idx = [*range(0, len(df), max(1, len(df) // n)), int(bottom.argmax()), int(bottom.argmin())]
    rows = df.iloc[idx]
    return [
        (IMAGES / f"{r.image_id}.jpg", (int(r.x), int(r.y), int(r.w), int(r.h)))
        for r in rows.itertuples(index=False)
    ]


@needs_images
def test_tensor_bitwise_equal_on_test_frames():
    for path, bbox in _test_rows():
        _assert_same(path, bbox)


@needs_images
def test_bbox_clipped_by_frame_edges_bitwise_equal():
    path, (x, y, w, h) = _test_rows(1)[0]
    for bbox in [(-20, y, w, h), (x, -5, w, 60), (x, 1000, w, 200), (1900, 1070, 100, 100)]:
        _assert_same(path, bbox)


@needs_images
def test_bbox_outside_frame_raises():
    path, _ = _test_rows(1)[0]
    for bbox in [(5000, 5000, 10, 10), (10, -50, 10, 20), (10, 1080, 10, 10)]:
        with pytest.raises(ValueError):
            read_crop(path, *bbox)


def test_non_rgb_and_non_jpeg_fall_back_to_full_decode(tmp_path):
    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, (120, 90, 3), dtype=np.uint8))
    cases = {
        "gray.jpg": rgb.convert("L"),
        "rgb.png": rgb,
        "rgb.jpg": rgb,
        "cmyk.jpg": rgb.convert("CMYK"),
    }
    for name, img in cases.items():
        path = tmp_path / name
        img.save(path, quality=90)
        _assert_same(path, (7, 11, 50, 70))
        _assert_same(path, (7, 11, 50, 200))


def test_truncated_jpeg_raises(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "cut.jpg"
    Image.fromarray(rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)).save(path, quality=95)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 3])
    with pytest.raises(OSError):
        read_crop(path, 0, 0, 300, 390)
