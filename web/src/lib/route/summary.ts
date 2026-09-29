import { msk } from "./time";
import type { Camera, Sighting } from "./types";

export interface CityStat {
  city: string;
  count: number;
  from: number;
  to: number;
}

/** Города по числу появлений. */
export function cities(accepted: Sighting[], cameras: Map<string, Camera>): CityStat[] {
  const by = new Map<string, CityStat>();
  for (const s of accepted) {
    const city = cameras.get(s.cameraId)?.city ?? "—";
    const c = by.get(city) ?? { city, count: 0, from: s.ts, to: s.ts };
    c.count++;
    c.from = Math.min(c.from, s.ts);
    c.to = Math.max(c.to, s.ts);
    by.set(city, c);
  }
  return [...by.values()].sort((a, b) => b.count - a.count);
}

/** Когда машину видят: [день недели, 0 — пн][час по Москве] → число появлений. */
export function heat(accepted: Sighting[]): number[][] {
  const grid = Array.from({ length: 7 }, () => Array<number>(24).fill(0));
  for (const s of accepted) {
    const p = msk(s.ts);
    grid[p.weekday][p.hour]++;
  }
  return grid;
}
