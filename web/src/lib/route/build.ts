import type { SearchResponse } from "../../api/types";
import { clusterPlaces, hypotheses, type PlaceHypothesis } from "./places";
import { REGISTRY } from "./registry";
import { addLookalike, hashSeed, simulateHistory } from "./simulate";
import { cities, heat, type CityStat } from "./summary";
import { ANCHOR, DAY, PERIOD_DAYS } from "./time";
import { buildTrack, type Track } from "./track";
import type { Camera, RealFrame, Registry } from "./types";

export interface RouteData {
  vehicleId: string;
  cameras: Map<string, Camera>;
  track: Track;
  places: PlaceHypothesis[];
  cities: CityStat[];
  heat: number[][];
  from: number;
  to: number;
  stats: { sightings: number; real: number; regular: number; cities: number; rejected: number };
}

/**
 * Камера кадра — одна из дорожных камер Москвы реестра, по хешу image_id. camera_id
 * датасета не используем: он анонимный, без геопривязки и времени, и на инференсе
 * недоступен (ответ жюри QО-1); по ТЗ §5.2 камера, время и гео участникам не передаются.
 */
export function cameraForImage(imageId: string, registry: Registry = REGISTRY): string {
  const road = registry.cameras.filter((c) => c.kind === "road" && c.city === "Москва");
  return road[hashSeed(imageId) % road.length].id;
}

/** Машину не удалось связать с кадрами — показать оператору, а не пустую карту. */
export class RouteError extends Error {}

/**
 * Ответ поиска (топ-100) → маршрут. Реальны: кадры с тем же vehicle_id (подтверждённая
 * идентичность в галерее), кадр запроса и их уверенность. Камера кадра смоделирована
 * (cameraForImage), остальное — тоже (simulate.ts).
 */
export function buildRoute(
  vehicleId: string,
  result: SearchResponse,
  queryImageId: string | null,
  registry: Registry = REGISTRY,
): RouteData {
  const frames: RealFrame[] = [];
  if (queryImageId) {
    const cameraId = cameraForImage(queryImageId, registry);
    frames.push({ cameraId, cropUrl: result.queryCropUrl, imageId: queryImageId, isQuery: true, vehicleId });
  }
  for (const c of result.candidates) {
    if (c.item.vehicleId !== vehicleId) continue;
    frames.push({ cameraId: cameraForImage(c.item.imageId, registry), cropUrl: c.item.cropUrl, galleryId: c.item.id, imageId: c.item.imageId, vehicleId, confidence: c.confidence });
  }
  if (!frames.some((f) => !f.isQuery)) {
    throw new RouteError(`В топ-100 по этому запросу нет кадров машины ${vehicleId}.`);
  }

  let history = simulateHistory(vehicleId, frames, registry);
  const other = result.candidates.find((c) => c.item.vehicleId !== vehicleId);
  if (other) {
    history = addLookalike(history, {
      cameraId: "",
      cropUrl: other.item.cropUrl,
      galleryId: other.item.id,
      imageId: other.item.imageId,
      vehicleId: other.item.vehicleId,
      confidence: other.confidence,
    }, registry);
  }

  const cameras = new Map(registry.cameras.map((c) => [c.id, c]));
  const track = buildTrack(history, cameras);
  const places = hypotheses(clusterPlaces(track.accepted, cameras));
  const cityStats = cities(track.accepted, cameras);
  return {
    vehicleId,
    cameras,
    track,
    places,
    cities: cityStats,
    heat: heat(track.accepted),
    from: ANCHOR - PERIOD_DAYS * DAY,
    to: ANCHOR,
    stats: {
      sightings: track.accepted.length,
      real: track.accepted.filter((s) => s.kind === "real").length,
      regular: places.filter((p) => p.role === "night" || p.role === "day").length,
      cities: cityStats.length,
      rejected: track.rejected.length,
    },
  };
}
