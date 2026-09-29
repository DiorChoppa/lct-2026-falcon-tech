import type { Bbox } from "../api/types";

/** Размер отображённого изображения и его исходный размер в пикселях. */
export interface Scale {
  displayW: number;
  displayH: number;
  naturalW: number;
  naturalH: number;
}

/** Точка на экране (относительно картинки) → пиксели исходного кадра, в границах кадра. */
export function toNatural(px: number, py: number, s: Scale): { x: number; y: number } {
  const x = Math.round((px * s.naturalW) / s.displayW);
  const y = Math.round((py * s.naturalH) / s.displayH);
  return { x: clamp(x, 0, s.naturalW), y: clamp(y, 0, s.naturalH) };
}

/** Прямоугольник из двух углов в любом порядке. */
export function fromCorners(
  a: { x: number; y: number },
  b: { x: number; y: number },
): Bbox {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return { x, y, w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
}

/** bbox в пикселях кадра → CSS-координаты поверх отображённой картинки. */
export function toDisplay(b: Bbox, s: Scale): Bbox {
  const kx = s.displayW / s.naturalW;
  const ky = s.displayH / s.naturalH;
  return { x: b.x * kx, y: b.y * ky, w: b.w * kx, h: b.h * ky };
}

/** Обрезка по кадру и защита от нулевой площади (правило api: 422 на bbox вне кадра). */
export function clampBbox(b: Bbox, naturalW: number, naturalH: number): Bbox {
  const x = clamp(b.x, 0, Math.max(naturalW - 1, 0));
  const y = clamp(b.y, 0, Math.max(naturalH - 1, 0));
  return {
    x,
    y,
    w: clamp(b.w, 1, naturalW - x),
    h: clamp(b.h, 1, naturalH - y),
  };
}

export function isUsable(b: Bbox | null): b is Bbox {
  return !!b && b.w >= 8 && b.h >= 8;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}
