import { describe, expect, it } from "vitest";
import { clusterPlaces, confidence, coveredDates, hypotheses, meanClock } from "./places";
import type { Camera, Sighting } from "./types";

const P = (id: string, lat: number, city = "Москва"): Camera => ({ id, kind: "parking", lat, lon: 37.62, address: `адрес ${id}`, city });
const cameras = new Map([P("home", 55.70), P("work", 55.75), P("mall", 55.80), P("tver", 56.86, "Тверь")].map((c) => [c.id, c]));
const at = (iso: string) => Date.parse(`${iso}:00+03:00`);
let n = 0;
const stay = (cameraId: string, iso: string, hours: number): Sighting => ({
  id: String(n++), cameraId, ts: at(iso), kind: "sim", cropUrl: "", dwellMin: hours * 60,
});

describe("ночи и будни", () => {
  it("стоянка 22:00–07:00 покрывает ночь следующей даты", () => {
    expect(coveredDates(stay("home", "2026-09-21T22:00", 9), 3, false)).toEqual(["2026-09-22"]);
  });
  it("09:00–18:00 во вторник — будний день, в субботу — нет", () => {
    expect(coveredDates(stay("work", "2026-09-22T09:00", 9), 13, true)).toEqual(["2026-09-22"]);
    expect(coveredDates(stay("work", "2026-09-26T09:00", 9), 13, true)).toEqual([]);
  });
});

describe("уверенность", () => {
  it("14 из 17 ночей ≈ 0,77", () => expect(confidence(14, 17)).toBeCloseTo(0.7735, 3));
  it("не выше 0,95", () => expect(confidence(30, 30)).toBe(0.95));
  it("без наблюдений — 0", () => expect(confidence(0, 0)).toBe(0));
});

describe("среднее время суток", () => {
  it("через полночь: 23:00 и 01:00 → 00:00", () => expect(meanClock([23 * 60, 60])).toBe("00:00"));
  it("днём: 09:00 и 10:00 → 09:30", () => expect(meanClock([9 * 60, 10 * 60])).toBe("09:30"));
});

describe("места и гипотезы", () => {
  const days = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"];
  const stays = [
    ...days.map((d) => stay("home", `${d}T22:30`, 9)),
    ...days.map((d) => stay("work", `${d}T09:00`, 8)),
    stay("mall", "2026-09-19T12:00", 2),
    stay("tver", "2026-09-20T14:00", 20),
  ];

  it("разные точки — разные места", () => {
    expect(clusterPlaces(stays, cameras)).toHaveLength(4);
  });

  it("ночная, дневная, поездка, эпизодическое", () => {
    const h = hypotheses(clusterPlaces(stays, cameras));
    const byPlace = new Map(h.map((x) => [x.place.address, x]));
    const home = byPlace.get("адрес home")!;
    expect(home.role).toBe("night");
    expect(home.guess).toBe("вероятно, место жительства");
    expect(home.explain).toMatch(/^5 из 6 наблюдённых ночей, в среднем 22:30–07:30$/);
    expect(byPlace.get("адрес work")!.role).toBe("day");
    expect(byPlace.get("адрес tver")!.role).toBe("trip");
    expect(byPlace.get("адрес tver")!.explain).toBe("вс, 20 сент. — пн, 21 сент. · 1 визит, 20 ч");
    expect(byPlace.get("адрес mall")!.role).toBe("episodic");
    expect(h[0].role).toBe("night");
  });
});
