import type { RouteData } from "../../lib/route/build";
import { msk } from "../../lib/route/time";
import type { Sighting } from "../../lib/route/types";

export type LngLat = [number, number];
export type Scope = "moscow" | "all";

export interface MapPoint {
  id: string;
  lon: number;
  lat: number;
  kind: Sighting["kind"];
  /** Подпись для скринридера и подсказки. */
  label: string;
  count: number;
  /** Номер в треке выбранного дня. */
  order?: number;
}

export interface MapPlace {
  id: string;
  lon: number;
  lat: number;
  role: "night" | "day";
  confidence: number;
  title: string;
}

export interface MapLayers {
  points: MapPoint[];
  places: MapPlace[];
  rejected: MapPoint[];
  line: LngLat[] | null;
  /** [[minLon, minLat], [maxLon, maxLat]] */
  bounds: [LngLat, LngLat];
}

/**
 * Данные маршрута → слои карты. Без выбранного дня — одна метка на камеру с
 * числом появлений; с днём — появления этого дня по порядку и линия трека.
 */
export function toLayers(data: RouteData, opts: { scope: Scope; day: string | null }): MapLayers {
  const cam = (id: string) => data.cameras.get(id)!;
  let points: MapPoint[];
  let line: LngLat[] | null = null;
  if (opts.day) {
    const today = data.track.accepted.filter((s) => msk(s.ts).dateKey === opts.day);
    points = today.map((s, i) => ({
      id: s.id, lon: cam(s.cameraId).lon, lat: cam(s.cameraId).lat, kind: s.kind,
      label: cam(s.cameraId).address, count: 1, order: i + 1,
    }));
    line = points.length > 1 ? points.map((p) => [p.lon, p.lat] as LngLat) : null;
  } else {
    const by = new Map<string, MapPoint>();
    for (const s of data.track.accepted) {
      const c = cam(s.cameraId);
      const p = by.get(c.id) ?? { id: c.id, lon: c.lon, lat: c.lat, kind: "sim", label: c.address, count: 0 };
      p.count++;
      if (s.kind === "real") p.kind = "real";
      by.set(c.id, p);
    }
    points = [...by.values()];
  }
  const places: MapPlace[] = data.places
    .filter((h) => h.role === "night" || h.role === "day")
    .map((h) => ({ id: h.place.id, lon: h.place.lon, lat: h.place.lat, role: h.role as "night" | "day", confidence: h.confidence ?? 0, title: h.fact }));
  const rejected: MapPoint[] = data.track.rejected.map((r) => {
    const c = cam(r.sighting.cameraId);
    return { id: r.sighting.id, lon: c.lon, lat: c.lat, kind: "lookalike", label: c.address, count: 1 };
  });

  // Границы: день — его точки; «Москва» — точки в Москве; иначе — все. Отсеянная
  // похожая машина всегда в кадре (в режиме дня — если она в этот день).
  const all = data.track.accepted.map((s) => cam(s.cameraId));
  const inScope = opts.scope === "moscow" ? all.filter((c) => c.city === "Москва") : all;
  const basis = opts.day && points.length ? points : inScope.length ? inScope : all;
  const odd = data.track.rejected
    .filter((r) => !opts.day || msk(r.sighting.ts).dateKey === opts.day)
    .map((r) => cam(r.sighting.cameraId));
  return { points, places, rejected, line, bounds: boundsOf([...basis, ...odd]) };
}

function boundsOf(xs: { lon: number; lat: number }[]): [LngLat, LngLat] {
  const lons = xs.map((x) => x.lon);
  const lats = xs.map((x) => x.lat);
  // Одна точка — рамка ~1 км, чтобы карте было куда приблизить.
  const pad = xs.length > 1 ? 0 : 0.01;
  return [
    [Math.min(...lons) - pad, Math.min(...lats) - pad],
    [Math.max(...lons) + pad, Math.max(...lats) + pad],
  ];
}
