"""Эталонный препроцессинг по манифесту models/model.json: кроп → тензор NCHW.

Один код для extractor.py (замер жюри), parity fixtures и валидации;
crates/inference повторяет его на Rust (tests/parity.rs).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

RESAMPLE = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}


def load_manifest(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def crop_bbox(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Кроп по bbox, обрезанный по границам кадра (как reid.data.load_crop, но из открытого кадра)."""
    left, top = max(int(x), 0), max(int(y), 0)
    right, bottom = min(int(x + w), img.width), min(int(y + h), img.height)
    if right <= left or bottom <= top:
        raise ValueError(f"bbox вне кадра: {(x, y, w, h)} при {img.size}")
    return img.crop((left, top, right, bottom))


def _decode_jpeg_rows(img: Image.Image, data: bytes, rows: int) -> Image.Image:
    """Первые `rows` строк RGB-JPEG тем же декодером Pillow, без строк ниже.

    libjpeg отдаёт строки сверху вниз, поэтому они побитно равны полному декоду
    (tests/test_preprocess.py). Остановка до конца кадра даёт код -2 (jpeg_finish_decompress
    без последних строк); n >= 0 — данные кончились раньше, файл обрезан.
    """
    tile = img.tile[0]
    decoder = Image._getdecoder(img.mode, tile.codec_name, tile.args, img.decoderconfig)
    frame = Image.core.new(img.mode, (img.width, rows))
    decoder.setimage(frame, (0, 0, img.width, rows))
    try:
        n, err = decoder.decode(data)
    finally:
        decoder.cleanup()
    if n >= 0 or err not in (0, -2):
        raise OSError(f"JPEG не декодирован: n={n}, err={err}")
    return img._new(frame)


def read_crop(path: Path | str, x: int, y: int, w: int, h: int) -> Image.Image:
    """Чтение + декод + кроп по bbox → RGB; пиксели те же, что crop_bbox(полный кадр).

    Базовый RGB-JPEG декодируется только до нижней строки bbox: строки ниже не нужны,
    а декод 1080p — самая дорогая часть цикла на CPU (docs/speed.md).
    """
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as img:
        rows = min(int(y + h), img.height)
        if (
            img.format == "JPEG"
            and img.mode == "RGB"
            and len(img.tile) == 1
            and img.tile[0].codec_name == "jpeg"
            and img.tile[0].offset == 0
            and 0 < rows < img.height
        ):
            frame = _decode_jpeg_rows(img, data, rows)
        else:
            frame = img if img.mode == "RGB" else img.convert("RGB")
            frame.load()
        return crop_bbox(frame, x, y, w, h)


def to_tensor(crop: Image.Image, m: dict) -> np.ndarray:
    """RGB-кроп целиком → resize по манифесту → (pixel/255 − mean) / std → CHW float32."""
    size = (m["input_width"], m["input_height"])
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    x = np.asarray(crop.resize(size, RESAMPLE[m["resize"]]), dtype=np.float32).transpose(2, 0, 1)
    # Те же операции float32 и в том же порядке, что (x / 255 − mean) / std, но сразу в CHW
    # и без промежуточных массивов: результат побитно прежний.
    out = np.empty(x.shape, dtype=np.float32)
    np.divide(x, 255.0, out=out)
    out -= np.asarray(m["mean"], dtype=np.float32)[:, None, None]
    out /= np.asarray(m["std"], dtype=np.float32)[:, None, None]
    return out
