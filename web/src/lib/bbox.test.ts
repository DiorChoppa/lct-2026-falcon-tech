import { describe, expect, it } from "vitest";
import { clampBbox, fromCorners, isUsable, toDisplay, toNatural } from "./bbox";

const scale = { displayW: 960, displayH: 540, naturalW: 1920, naturalH: 1080 };

describe("toNatural", () => {
  it("масштабирует экранные координаты в пиксели кадра", () => {
    expect(toNatural(480, 270, scale)).toEqual({ x: 960, y: 540 });
  });
  it("не выходит за кадр при перетаскивании за край картинки", () => {
    expect(toNatural(-20, 600, scale)).toEqual({ x: 0, y: 1080 });
  });
});

describe("fromCorners", () => {
  it("нормализует углы, нарисованные справа налево", () => {
    expect(fromCorners({ x: 300, y: 200 }, { x: 100, y: 50 })).toEqual({
      x: 100,
      y: 50,
      w: 200,
      h: 150,
    });
  });
});

describe("toDisplay", () => {
  it("обратен toNatural для целых коэффициентов", () => {
    expect(toDisplay({ x: 200, y: 100, w: 400, h: 300 }, scale)).toEqual({
      x: 100,
      y: 50,
      w: 200,
      h: 150,
    });
  });
});

describe("clampBbox", () => {
  it("обрезает bbox, вылезающий за кадр, не давая нулевой ширины", () => {
    expect(clampBbox({ x: 1900, y: 1070, w: 100, h: 100 }, 1920, 1080)).toEqual({
      x: 1900,
      y: 1070,
      w: 20,
      h: 10,
    });
  });
  it("оставляет корректный bbox как есть", () => {
    const b = { x: 10, y: 20, w: 300, h: 200 };
    expect(clampBbox(b, 1920, 1080)).toEqual(b);
  });
});

describe("isUsable", () => {
  it("отклоняет случайный клик без протяжки", () => {
    expect(isUsable({ x: 5, y: 5, w: 3, h: 2 })).toBe(false);
    expect(isUsable(null)).toBe(false);
    expect(isUsable({ x: 5, y: 5, w: 40, h: 30 })).toBe(true);
  });
});
