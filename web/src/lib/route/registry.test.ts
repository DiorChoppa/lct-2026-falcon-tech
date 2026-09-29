import { describe, expect, it } from "vitest";
import { CAMERAS_BY_ID, REGISTRY } from "./registry";

describe("реестр камер", () => {
  it("камеры 0–95 — дорожные, в Москве", () => {
    for (let i = 0; i < 96; i++) {
      const c = CAMERAS_BY_ID.get(String(i));
      expect(c?.kind).toBe("road");
      expect(c?.city).toBe("Москва");
    }
  });

  it("у генератора есть из чего выбрать дом, работу и досуг", () => {
    const zone = (z: string) => REGISTRY.cameras.filter((c) => c.zone === z).length;
    expect(zone("residential")).toBeGreaterThanOrEqual(5);
    expect(zone("business")).toBeGreaterThanOrEqual(5);
    expect(zone("leisure")).toBeGreaterThanOrEqual(3);
  });

  it("маршруты ссылаются на существующие камеры вне Москвы", () => {
    for (const r of REGISTRY.routes) {
      for (const id of [...r.cameraIds, r.parkingId]) {
        expect(CAMERAS_BY_ID.get(id)?.city).not.toBe("Москва");
      }
      expect(CAMERAS_BY_ID.get(r.parkingId)?.kind).toBe("parking");
    }
  });
});
