import { describe, expect, it } from "vitest";
import type { Candidate, SearchResponse } from "../../api/types";
import { buildRoute } from "../../lib/route/build";
import { REGISTRY } from "../../lib/route/registry";
import { msk } from "../../lib/route/time";
import { toLayers } from "./layers";

const cand = (id: number, imageId: string, vehicleId: string, confidence: number): Candidate => ({
  item: { id, imageId, vehicleId, bbox: { x: 0, y: 0, w: 10, h: 10 }, cropUrl: `/c/${id}.jpg`, plate: null, tags: [], createdAt: "" },
  score: confidence, confidence, localScore: null, accepted: true,
});
const res: SearchResponse = { id: 1, queryCropUrl: "/q.jpg", threshold: 0.661, candidates: [cand(1, "g1111111", "959", 0.9), cand(2, "x9999999", "606", 0.6)] };
const data = buildRoute("959", res, "q0000000", REGISTRY);

describe("слои карты", () => {
  it("без выбранного дня — одна метка на камеру", () => {
    const l = toLayers(data, { scope: "all", day: null });
    expect(l.points).toHaveLength(new Set(data.track.accepted.map((s) => s.cameraId)).size);
    expect(l.line).toBeNull();
    expect(l.rejected).toHaveLength(1);
  });

  it("выбранный день — метки по порядку и линия трека", () => {
    const day = msk(data.track.accepted.find((s) => s.kind === "real")!.ts).dateKey;
    const l = toLayers(data, { scope: "all", day });
    expect(l.points.every((p) => p.order !== undefined)).toBe(true);
    expect(l.line?.length).toBe(l.points.length);
  });

  it("день без появлений — ни меток, ни линии, границы остаются", () => {
    const l = toLayers(data, { scope: "all", day: "2020-01-01" });
    expect(l.points).toEqual([]);
    expect(l.line).toBeNull();
    expect(l.bounds[0][0]).toBeLessThan(l.bounds[1][0]);
  });

  it("отсеянная похожая машина — в границах обоих охватов и своего дня", () => {
    const look = data.track.rejected[0].sighting;
    const c = data.cameras.get(look.cameraId)!;
    const inside = ([[minLon, minLat], [maxLon, maxLat]]: [number, number][]) =>
      c.lon >= minLon && c.lon <= maxLon && c.lat >= minLat && c.lat <= maxLat;
    for (const scope of ["moscow", "all"] as const) {
      expect(inside(toLayers(data, { scope, day: null }).bounds)).toBe(true);
      expect(inside(toLayers(data, { scope, day: msk(look.ts).dateKey }).bounds)).toBe(true);
    }
  });

  it("охват «Москва» не включает Петербург и другие города", () => {
    const [[, minLat], [, maxLat]] = toLayers(data, { scope: "moscow", day: null }).bounds;
    expect(minLat).toBeGreaterThan(55.4);
    expect(maxLat).toBeLessThan(56.1);
  });
});
