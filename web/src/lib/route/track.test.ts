import { describe, expect, it } from "vitest";
import { ANCHOR, MIN } from "./time";
import { buildTrack } from "./track";
import type { Camera, Sighting } from "./types";

const cam = (id: string, kmNorth: number): Camera => ({
  id, kind: "road", lat: 55.75 + kmNorth / 111.195, lon: 37.62, address: id, city: "Москва",
});
const cameras = new Map([cam("a", 0), cam("b", 5), cam("far", 40)].map((c) => [c.id, c]));
const s = (id: string, cameraId: string, minute: number, kind: Sighting["kind"] = "sim", confidence?: number): Sighting => ({
  id, cameraId, ts: ANCHOR + minute * MIN, kind, cropUrl: "", confidence,
});

describe("сборка трека", () => {
  it("совместимые появления принимаются все, по времени", () => {
    const t = buildTrack([s("2", "b", 30), s("1", "a", 0)], cameras);
    expect(t.accepted.map((x) => x.id)).toEqual(["1", "2"]);
    expect(t.rejected).toEqual([]);
  });

  it("двойник далеко и сразу после реального кадра отсеивается с причиной", () => {
    const real = s("r", "a", 0, "real", 0.9);
    const look = s("l", "far", 9, "lookalike", 0.66);
    const t = buildTrack([look, real], cameras);
    expect(t.accepted.map((x) => x.id)).toEqual(["r"]);
    expect(t.rejected).toHaveLength(1);
    expect(t.rejected[0].sighting.id).toBe("l");
    expect(t.rejected[0].conflictWith.id).toBe("r");
    expect(t.rejected[0].needKmh).toBeGreaterThan(200);
  });

  it("при конфликте уступает менее надёжное: смоделированное — реальному", () => {
    const t = buildTrack([s("sim", "far", 5), s("real", "a", 0, "real", 0.9)], cameras);
    expect(t.accepted.map((x) => x.id)).toEqual(["real"]);
    expect(t.rejected[0].sighting.id).toBe("sim");
  });

  it("появление на неизвестной камере пропускается", () => {
    const t = buildTrack([s("x", "nope", 0)], cameras);
    expect(t.accepted).toEqual([]);
    expect(t.rejected).toEqual([]);
  });
});
