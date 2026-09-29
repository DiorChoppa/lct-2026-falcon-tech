"""Поиск анонимизированного номера на кропе ТС — для контрольного прогона «номер залит».

Номера на кадрах датасета закрыты пикселизацией: плоские блоки (соседние пиксели
равны с точностью ±2) со ступеньками на общей сетке. Плоское стекло или краска дают
плоскость, но не сетку ступенек — по ней и отличаем. Координат номеров организаторы
не передают (ответ 48), поэтому это приближение их контрольной версии теста.
Не участвует в инференсе: только диагностика `scripts/plate_control.py`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 в координатах кропа, x1/y1 исключительно


def _flat(a: np.ndarray) -> np.ndarray:
    eqx = np.all(np.abs(a[:, 1:] - a[:, :-1]) <= 2, axis=2)
    eqy = np.all(np.abs(a[1:, :] - a[:-1, :]) <= 2, axis=2)
    f = np.zeros(a.shape[:2], bool)
    f[:-1, :-1] = eqx[:-1, :] & eqy[:, :-1]
    return f


def _grid(gray: np.ndarray) -> tuple[float, float, float]:
    """(доля плоских переходов, доля ступенек, выровненность ступенек по линиям сетки)."""
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    flat = (np.mean(dx <= 1) + np.mean(dy <= 1)) / 2
    steps = (np.mean(dx >= 5) + np.mean(dy >= 5)) / 2

    def align(s: np.ndarray, axis: int) -> float:
        frac = s.mean(axis=axis)  # у пикселизации ступенька тянется через весь блок
        return float((frac**2).sum() / frac.sum()) if frac.sum() else 0.0

    return float(flat), float(steps), min(align(dx >= 5, 0), align(dy >= 5, 1))


def plate_boxes(rgb: np.ndarray, pad: float = 0.12) -> list[Box]:
    """Все пикселизованные области формы номера в нижних 70 % кропа, с запасом по краям."""
    a = rgb.astype(np.int16)
    H, W = a.shape[:2]
    lum = a.mean(axis=2)
    m = _flat(a) & (lum > 60) & (lum < 245)
    m = ndimage.binary_opening(m, np.ones((3, 3), bool))
    m = ndimage.binary_closing(m, np.ones((5, 11), bool))
    labels, _ = ndimage.label(m, np.ones((3, 3), bool))
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        area = int((labels[sl] == i).sum())
        if not (0.05 * W <= w <= 0.45 * W and 6 <= h <= 0.14 * H and 1.3 <= w / h <= 8):
            continue
        if area / (w * h) < 0.45 or (y0 + y1) / 2 < 0.3 * H:
            continue
        flat, steps, align = _grid(lum[y0:y1, x0:x1])
        if flat < 0.55 or steps < 0.015 or align < 0.4:
            continue
        px, py = pad * w, pad * 2 * h
        out.append(
            (int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py)))
        )
    return out


def control_boxes(boxes: list[Box], size: tuple[int, int], seed: int) -> list[Box]:
    """Прямоугольники того же размера в случайном месте кропа, не задевающие номер."""
    W, H = size
    rng = np.random.default_rng(seed)
    out = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        for _ in range(100):
            cx, cy = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
            c = (cx, cy, cx + w, cy + h)
            if all(c[2] <= b[0] or c[0] >= b[2] or c[3] <= b[1] or c[1] >= b[3] for b in boxes):
                out.append(c)
                break
    return out
