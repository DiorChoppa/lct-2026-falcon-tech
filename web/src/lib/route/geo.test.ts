import { describe, expect, it } from "vitest";
import { feasibility, haversineKm } from "./geo";
import { ANCHOR, HOUR, MIN, msk, mskMidnight } from "./time";
import type { Camera, Sighting } from "./types";

const KM_PER_DEG = 111.195;
const cam = (id: string, kmNorth: number, city = "Москва"): Camera => ({
  id, kind: "road", lat: 55.75 + kmNorth / KM_PER_DEG, lon: 37.62, address: id, city,
});
const at = (id: string, cameraId: string, minute: number, extra: Partial<Sighting> = {}): Sighting => ({
  id, cameraId, ts: ANCHOR + minute * MIN, kind: "sim", cropUrl: "", ...extra,
});

describe("время", () => {
  it("конец периода — четверг 24.09.2026, полночь по Москве", () => {
    expect(msk(ANCHOR)).toEqual({ dateKey: "2026-09-24", weekday: 3, hour: 0, minute: 0 });
    expect(mskMidnight(ANCHOR + 5 * HOUR)).toBe(ANCHOR);
  });
});

describe("гаверсинус", () => {
  it("Москва — Санкт-Петербург ≈ 634 км", () => {
    const km = haversineKm({ lat: 55.7558, lon: 37.6173 }, { lat: 59.9343, lon: 30.3351 });
    expect(Math.abs(km - 634)).toBeLessThan(5);
  });
});

describe("успел ли доехать", () => {
  it("10 км за 20 минут — успел", () => {
    const a = cam("a", 0), b = cam("b", 10);
    expect(feasibility(at("1", "a", 0), a, at("2", "b", 20), b).ok).toBe(true);
  });

  it("38 км за 9 минут — нет, нужна скорость ≈ 330 км/ч", () => {
    const a = cam("a", 0), b = cam("b", 38);
    const f = feasibility(at("1", "a", 0), a, at("2", "b", 9), b);
    expect(f.ok).toBe(false);
    expect(f.needKmh).toBeCloseTo(329.3, 0);
  });

  it("граница допуска: без паузы доступно 15 км по дорогам (90 км/ч × 10 мин)", () => {
    const a = cam("a", 0);
    expect(feasibility(at("1", "a", 0), a, at("2", "b", 0), cam("b", 11.5)).ok).toBe(true);
    expect(feasibility(at("1", "a", 0), a, at("2", "c", 0), cam("c", 11.6)).ok).toBe(false);
  });

  it("стоянка занимает время: отъезд считается от её конца", () => {
    const a = cam("a", 0), b = cam("b", 20);
    const parked = at("1", "a", 0, { dwellMin: 60 });
    expect(feasibility(parked, a, at("2", "b", 40), b).ok).toBe(false);
    expect(feasibility(parked, a, at("3", "b", 90), b).ok).toBe(true);
  });

  it("за городом предел 130 км/ч", () => {
    const a = cam("a", 0, "Тверь"), b = cam("b", 49, "Тверь");
    expect(feasibility(at("1", "a", 0), a, at("2", "b", 20), b).ok).toBe(true);
    const c = cam("c", 49);
    expect(feasibility(at("1", "a0", 0), cam("a0", 0), at("2", "c", 20), c).ok).toBe(false);
  });

  it("та же камера совместима всегда", () => {
    const a = cam("a", 0);
    expect(feasibility(at("1", "a", 0, { dwellMin: 600 }), a, at("2", "a", 5), a).ok).toBe(true);
  });
});
