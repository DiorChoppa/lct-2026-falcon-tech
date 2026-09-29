import { describe, expect, it } from "vitest";
import { CAMERAS_BY_ID, REGISTRY } from "./registry";
import { addLookalike, simulateHistory } from "./simulate";
import { ANCHOR, DAY, PERIOD_DAYS } from "./time";
import { buildTrack } from "./track";
import type { RealFrame } from "./types";

const real: RealFrame[] = [
  { cameraId: "22", cropUrl: "/q.jpg", isQuery: true, vehicleId: "959" },
  { cameraId: "61", cropUrl: "/g1.jpg", galleryId: 1, confidence: 0.91, vehicleId: "959" },
  { cameraId: "61", cropUrl: "/g2.jpg", galleryId: 2, confidence: 0.88, vehicleId: "959" },
];
const look: RealFrame = { cameraId: "", cropUrl: "/other.jpg", galleryId: 9, confidence: 0.65, vehicleId: "1" };
// Как у 1115 на стенде: кадр запроса и 7 кадров галереи на шести камерах.
const many: RealFrame[] = ["94", "19", "10", "40", "40", "6", "6", "46"].map((cameraId, i) => ({
  cameraId, cropUrl: `/m${i}.jpg`, isQuery: i === 1, galleryId: i, confidence: 0.8, vehicleId: "1115",
}));

describe("генератор истории", () => {
  it("детерминирован по vehicle_id", () => {
    expect(simulateHistory("959", real, REGISTRY)).toEqual(simulateHistory("959", real, REGISTRY));
    expect(simulateHistory("959", real, REGISTRY)).not.toEqual(simulateHistory("1115", real, REGISTRY));
  });

  it("все реальные кадры встают в историю", () => {
    const h = simulateHistory("959", real, REGISTRY);
    expect(h.filter((s) => s.kind === "real")).toHaveLength(real.length);
    expect(h.find((s) => s.isQuery)?.cameraId).toBe("22");
  });

  it.each(["959", "1115", "1420", "7", "2024"])("укладывается в период и объём 80–130 появлений (машина %s)", (v) => {
    for (const frames of [real, many]) {
      const h = simulateHistory(v, frames, REGISTRY);
      expect(h.length).toBeGreaterThanOrEqual(80);
      expect(h.length).toBeLessThanOrEqual(130);
      for (const s of h) {
        expect(s.ts).toBeGreaterThanOrEqual(ANCHOR - PERIOD_DAYS * DAY);
        expect(s.ts).toBeLessThanOrEqual(ANCHOR);
      }
    }
  });

  it("не выходит за конец периода: ts ≤ ANCHOR, стоянки неотрицательны", () => {
    // машина 112: дом и работа далеко — вечер последнего дня раньше уходил за полночь
    const far: RealFrame[] = [
      { cameraId: "24", cropUrl: "/q.jpg", isQuery: true, vehicleId: "112" },
      { cameraId: "51", cropUrl: "/g1.jpg", galleryId: 1, confidence: 0.9, vehicleId: "112" },
      { cameraId: "51", cropUrl: "/g2.jpg", galleryId: 2, confidence: 0.9, vehicleId: "112" },
      { cameraId: "51", cropUrl: "/g3.jpg", galleryId: 3, confidence: 0.9, vehicleId: "112" },
    ];
    for (const v of ["112", "126", "148", "347", "440", "497"]) {
      for (const s of simulateHistory(v, far, REGISTRY)) {
        expect(s.ts).toBeLessThanOrEqual(ANCHOR);
        expect(s.dwellMin ?? 0).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it.each(["959", "1115", "1420", "7", "2024"])("своя история машины %s физически непротиворечива", (v) => {
    const t = buildTrack(simulateHistory(v, real, REGISTRY), CAMERAS_BY_ID);
    expect(t.rejected).toEqual([]);
  });

  it("у смоделированных событий — иллюстративный кроп этой же машины", () => {
    const h = simulateHistory("959", real, REGISTRY);
    const crops = new Set(real.map((f) => f.cropUrl));
    expect(h.filter((s) => s.kind === "sim").every((s) => crops.has(s.cropUrl))).toBe(true);
  });

  it("есть поездка за пределы Москвы", () => {
    const h = simulateHistory("959", real, REGISTRY);
    expect(h.some((s) => CAMERAS_BY_ID.get(s.cameraId)?.city !== "Москва")).toBe(true);
  });
});

describe("двойник", () => {
  it.each(["959", "1115", "1420", "7", "2024"])("всегда отсеивается проверкой (машина %s)", (v) => {
    const h = addLookalike(simulateHistory(v, real, REGISTRY), look, REGISTRY);
    const t = buildTrack(h, CAMERAS_BY_ID);
    expect(t.rejected.map((r) => r.sighting.kind)).toEqual(["lookalike"]);
    expect(t.rejected[0].conflictWith.kind).not.toBe("lookalike");
  });

  it.each(["959", "1115", "1420", "7", "2024"])("причина отказа правдоподобна: сразу после реального кадра (машина %s)", (v) => {
    for (const frames of [real, many]) {
      const [r] = buildTrack(addLookalike(simulateHistory(v, frames, REGISTRY), look, REGISTRY), CAMERAS_BY_ID).rejected;
      expect(r.conflictWith.kind).toBe("real");
      expect(r.gapMin).toBeGreaterThan(0);
      expect(r.needKmh).toBeLessThan(400);
    }
  });

  it("без реальных кадров двойника некуда поставить — история не меняется", () => {
    const h = simulateHistory("959", [], REGISTRY);
    expect(addLookalike(h, look, REGISTRY)).toBe(h);
  });
});
