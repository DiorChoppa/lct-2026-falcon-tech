import { MIN } from "./time";
import type { Camera, Sighting } from "./types";

/** Дороги длиннее прямой: расстояние по сфере умножаем на коэффициент извилистости. */
export const ROAD_FACTOR = 1.3;
/** Допуск на неточность часов камер и короткие остановки. */
export const SLACK_MIN = 10;
const EARTH_KM = 6371;

export function haversineKm(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_KM * Math.asin(Math.sqrt(h));
}

/** 90 км/ч между камерами Москвы, 130 — если хоть одна за городом. */
export function speedLimitKmh(a: Camera, b: Camera): number {
  return a.city === "Москва" && b.city === "Москва" ? 90 : 130;
}

/** Когда машина покинула место появления (для стоянок — конец стоянки). */
export function endTs(s: Sighting): number {
  return s.ts + (s.dwellMin ?? 0) * MIN;
}

export interface Feasibility {
  ok: boolean;
  /** Расстояние по дорогам (× ROAD_FACTOR). */
  km: number;
  /** Минут от конца a до начала b; отрицательно, если a ещё стоит. */
  gapMin: number;
  /** Скорость, которая понадобилась бы. */
  needKmh: number;
}

/** Могла ли машина из появления a (более раннего) успеть к появлению b. */
export function feasibility(a: Sighting, ca: Camera, b: Sighting, cb: Camera): Feasibility {
  const km = a.cameraId === b.cameraId ? 0 : haversineKm(ca, cb) * ROAD_FACTOR;
  const gapMin = (b.ts - endTs(a)) / MIN;
  const needKmh = km === 0 ? 0 : (km / Math.max(gapMin, 0.5)) * 60;
  const ok = km <= (speedLimitKmh(ca, cb) * Math.max(gapMin + SLACK_MIN, 0)) / 60;
  return { ok, km, gapMin, needKmh };
}
