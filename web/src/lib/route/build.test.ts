import { describe, expect, it } from "vitest";
import type { Candidate, SearchResponse } from "../../api/types";
import { buildRoute, cameraForImage, RouteError } from "./build";
import { REGISTRY } from "./registry";
import { heat } from "./summary";

const cand = (id: number, imageId: string, vehicleId: string | null, confidence: number): Candidate => ({
  item: { id, imageId, vehicleId, bbox: { x: 0, y: 0, w: 10, h: 10 }, cropUrl: `/c/${id}.jpg`, plate: null, tags: [], createdAt: "" },
  score: confidence, confidence, localScore: null, accepted: confidence >= 0.66,
});
const result = (candidates: Candidate[]): SearchResponse => ({ id: 1, queryCropUrl: "/q.jpg", threshold: 0.661, candidates });

describe("камера кадра", () => {
  it("смоделирована: детерминированно, дорожная камера Москвы, для любого image_id", () => {
    for (const id of ["6774d18f78dd4198a14977ec60ec0c3c", "не-из-датасета", "q0000000dddd"]) {
      const cam = cameraForImage(id);
      expect(cameraForImage(id)).toBe(cam);
      const c = REGISTRY.cameras.find((x) => x.id === cam);
      expect(c?.kind).toBe("road");
      expect(c?.city).toBe("Москва");
    }
  });
});

describe("сборка маршрута", () => {
  const r = buildRoute("959", result([
    cand(1, "g1111111aaaa", "959", 0.91),
    cand(2, "x9999999bbbb", "606", 0.65),
    cand(3, "g2222222cccc", "959", 0.88),
  ]), "q0000000dddd", REGISTRY);

  it("реальные кадры: запрос и два кадра галереи", () => {
    const real = r.track.accepted.filter((s) => s.kind === "real");
    expect(real).toHaveLength(3);
    expect(real.filter((s) => s.isQuery)).toHaveLength(1);
  });

  it("чужая машина из топа — двойник в отсеянном", () => {
    expect(r.track.rejected.map((x) => x.sighting.vehicleId)).toEqual(["606"]);
    expect(r.stats.rejected).toBe(1);
  });

  it("есть регулярная ночная стоянка и сводки согласованы", () => {
    expect(r.places.some((p) => p.role === "night")).toBe(true);
    expect(r.stats.sightings).toBe(r.track.accepted.length);
    expect(heat(r.track.accepted).flat().reduce((a, b) => a + b, 0)).toBe(r.track.accepted.length);
    expect(r.cities.reduce((a, c) => a + c.count, 0)).toBe(r.track.accepted.length);
  });

  it("без чужих кандидатов двойника нет", () => {
    const only = buildRoute("959", result([cand(1, "g1111111aaaa", "959", 0.91)]), null, REGISTRY);
    expect(only.track.rejected).toEqual([]);
  });

  it("машины нет в топе — понятная ошибка", () => {
    expect(() => buildRoute("959", result([cand(2, "x9999999bbbb", "606", 0.65)]), null, REGISTRY)).toThrow(RouteError);
  });
});
