"""Кропы ТС из полных кадров по bbox из CSV."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_crop(path: Path | str, x: int, y: int, w: int, h: int) -> Image.Image:
    """Кроп по bbox (x, y, w, h в пикселях кадра), обрезанный по границам кадра, RGB.

    Это эталон препроцессинга: Rust-инференс должен резать так же (тест паритета).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
    left, top = max(int(x), 0), max(int(y), 0)
    right, bottom = min(int(x + w), img.width), min(int(y + h), img.height)
    if right <= left or bottom <= top:
        raise ValueError(f"bbox вне кадра: {path} {(x, y, w, h)}")
    return img.crop((left, top, right, bottom))
