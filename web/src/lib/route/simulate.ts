import { haversineKm, ROAD_FACTOR, SLACK_MIN } from "./geo";
import { ANCHOR, DAY, MIN, msk, PERIOD_DAYS } from "./time";
import type { Camera, RealFrame, Registry, Sighting } from "./types";

// Смоделированная история машины за 30 дней (спецификация §3.3). Всё
// детерминировано зерном от vehicle_id: та же машина — та же история.

/** FNV-1a: зерно генератора из строки. */
export function hashSeed(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** mulberry32: маленький детерминированный ГПСЧ, значения в [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Средние скорости генератора заметно ниже пределов проверки (90/130 км/ч):
// собственная история машины всегда физически непротиворечива.
const CITY_KMH = 30;
const HIGHWAY_KMH = 75;
/** Доля событий, которые камеры «не зафиксировали». */
const DROP = 0.25;

interface Draft {
  cameraId: string;
  ts: number;
  dwellMin?: number;
}

export function simulateHistory(vehicleId: string, real: RealFrame[], registry: Registry, anchor = ANCHOR): Sighting[] {
  const rng = mulberry32(hashSeed(vehicleId));
  const pick = <T,>(xs: T[]): T => xs[Math.floor(rng() * xs.length)];
  const between = (lo: number, hi: number) => lo + rng() * (hi - lo);
  const byId = new Map(registry.cameras.map((c) => [c.id, c]));
  const moscow = registry.cameras.filter((c) => c.city === "Москва");
  const zone = (z: Camera["zone"]) => moscow.filter((c) => c.kind === "parking" && c.zone === z);
  const nearest = (xs: Camera[], to: Camera) => [...xs].sort((a, b) => haversineKm(a, to) - haversineKm(b, to))[0];

  // Дом и работа — рядом с реальными камерами машины: реальные кадры
  // естественно ложатся в поездки между ними.
  const realCams = [...new Set(real.map((f) => f.cameraId))]
    .map((id) => byId.get(id))
    .filter((c): c is Camera => !!c && c.kind === "road");
  const home = realCams.length ? nearest(zone("residential"), realCams[0]) : pick(zone("residential"));
  const work = realCams.length ? nearest(zone("business"), realCams[realCams.length - 1]) : pick(zone("business"));
  const leisure = zone("leisure");
  const trip = pick(registry.routes);
  const tripPath = trip.cameraIds.map((id) => byId.get(id)!);
  const tripDest = byId.get(trip.parkingId)!;

  // Дорога дом → работа: реальные камеры машины и ещё до двух с наименьшим крюком.
  const detour = (c: Camera) => haversineKm(home, c) + haversineKm(c, work) - haversineKm(home, work);
  const extra = moscow
    .filter((c) => c.kind === "road" && !realCams.includes(c))
    .sort((a, b) => detour(a) - detour(b))
    .slice(0, Math.max(0, 2 - realCams.length));
  const byHome = (a: Camera, b: Camera) => haversineKm(home, a) - haversineKm(home, b);
  const commute = [...realCams, ...extra].sort(byHome);
  // Обычный будний день — две камеры с наименьшим крюком: объём истории не растёт
  // с числом реальных камер; мимо всех реальных камер машина едет в одно утро.
  const daily = [...commute].sort((a, b) => detour(a) - detour(b)).slice(0, 2).sort(byHome);

  const out: Draft[] = [];
  const travel = (a: Camera, b: Camera, kmh: number) =>
    Math.max(2, ((haversineKm(a, b) * ROAD_FACTOR) / kmh) * 60) * MIN;
  const drive = (from: Camera, path: Camera[], to: Camera, t: number, kmh: number) => {
    let pos = from;
    for (const c of path) {
      t += travel(pos, c, kmh);
      out.push({ cameraId: c.id, ts: t });
      pos = c;
    }
    return t + travel(pos, to, kmh);
  };
  const stay = (c: Camera, from: number, to: number) =>
    out.push({ cameraId: c.id, ts: from, dwellMin: Math.round((to - from) / MIN) });

  const start = anchor - PERIOD_DAYS * DAY;
  const saturdays = [...Array(PERIOD_DAYS - 1).keys()].filter((d) => msk(start + d * DAY).weekday === 5);
  const tripDay = pick(saturdays);
  const weekdays = [...Array(PERIOD_DAYS).keys()].filter((d) => msk(start + d * DAY).weekday < 5);
  const realDay = pick(weekdays); // утро с реальными кадрами
  let realFrom = 0; // индекс в out первой камеры этого утра
  let homeSince = start; // ночь перед периодом — с начала периода

  for (let d = 0; d < PERIOD_DAYS; d++) {
    const day = start + d * DAY;
    if (d === tripDay) {
      const leave = day + between(8 * 60, 9 * 60) * MIN;
      stay(home, homeSince, leave);
      const arrive = drive(home, tripPath, tripDest, leave, HIGHWAY_KMH);
      const back = day + DAY + between(13 * 60, 15 * 60) * MIN;
      stay(tripDest, arrive, back);
      homeSince = drive(tripDest, [...tripPath].reverse(), home, back, HIGHWAY_KMH);
      d++; // воскресенье ушло на дорогу обратно
      continue;
    }
    if (msk(day).weekday < 5) {
      const leave = day + between(7 * 60 + 30, 8 * 60 + 40) * MIN;
      stay(home, homeSince, leave);
      if (d === realDay) realFrom = out.length;
      const atWork = drive(home, d === realDay ? commute : daily, work, leave, CITY_KMH);
      const off = atWork + between(7 * 60, 9 * 60) * MIN;
      stay(work, atWork, off);
      // вечером — одна камера из двух: обратно едут не всегда той же дорогой
      let t = drive(work, [pick(daily)], home, off, CITY_KMH);
      if (rng() < 0.3) {
        // вечером — в торговый центр
        const mall = pick(leisure);
        const at = t + travel(home, mall, CITY_KMH);
        const done = at + between(60, 120) * MIN;
        stay(mall, at, done);
        t = done + travel(mall, home, CITY_KMH);
      }
      homeSince = t;
    } else {
      const leave = day + between(11 * 60, 13 * 60) * MIN;
      stay(home, homeSince, leave);
      const place = pick(leisure);
      const at = leave + travel(home, place, CITY_KMH);
      const done = at + between(90, 180) * MIN;
      stay(place, at, done);
      homeSince = done + travel(place, home, CITY_KMH);
    }
  }
  // вечер последнего дня мог уйти за полночь — такой хвост за концом периода не пишем
  if (homeSince < anchor) stay(home, homeSince, anchor);

  // Реальные кадры — в одно из утр по дороге на работу, каждый на своей камере.
  const realAt = new Map<string, number>();
  for (let k = realFrom; k < realFrom + commute.length; k++) realAt.set(out[k].cameraId, out[k].ts);

  const history: Sighting[] = [];
  let n = 0;
  const nextId = () => `${vehicleId}-${n++}`;
  for (const e of out) {
    if (realAt.get(e.cameraId) === e.ts) continue; // это место займут реальные кадры
    if (rng() < DROP) continue;
    if (e.ts >= anchor) continue; // за концом периода
    const dwellMin = e.dwellMin === undefined ? undefined : Math.min(e.dwellMin, Math.round((anchor - e.ts) / MIN));
    history.push({ id: nextId(), cameraId: e.cameraId, ts: e.ts, kind: "sim", cropUrl: "", dwellMin });
  }
  real.forEach((f, j) => {
    const t = realAt.get(f.cameraId);
    if (t !== undefined) history.push({ ...f, id: nextId(), ts: t + j * 4000, kind: "real" });
  });

  // Иллюстративный кроп для смоделированных событий — по кругу из реальных кадров машины.
  const crops = real.map((f) => f.cropUrl);
  history.forEach((s, i) => {
    if (s.kind === "sim") s.cropUrl = crops.length ? crops[i % crops.length] : "";
  });
  return history.sort((a, b) => a.ts - b.ts);
}

/**
 * Двойник: реальный кадр похожей машины получает смоделированное появление,
 * физически несовместимое с реальным кадром нашей: через m минут после него,
 * пока следующего события нашей машины ещё нет, на камере Москвы, до которой
 * пришлось бы ехать как можно ближе к 250 км/ч (спецификация §3.4: «через 9 мин
 * в 38 км»). Отсеивается с запасом: дальше порога проверки (90 км/ч + допуск).
 */
export function addLookalike(history: Sighting[], frame: RealFrame, registry: Registry): Sighting[] {
  const roads = registry.cameras.filter((c) => c.city === "Москва" && c.kind === "road");
  let best: { ts: number; cameraId: string; off: number } | null = null;
  for (const [i, anchor] of history.entries()) {
    if (anchor.kind !== "real") continue;
    const from = registry.cameras.find((c) => c.id === anchor.cameraId)!;
    // окно — до следующего события истории: тогда проверка сравнит двойника именно с этим кадром
    const gapMin = ((history[i + 1]?.ts ?? Infinity) - anchor.ts) / MIN;
    for (let m = 1; m <= 9 && m < gapMin; m++) {
      for (const c of roads) {
        const km = haversineKm(from, c) * ROAD_FACTOR;
        if (km < ((90 * (m + SLACK_MIN)) / 60) * 1.1) continue;
        const off = Math.abs((km / m) * 60 - 250);
        if (!best || off < best.off) best = { ts: anchor.ts + m * MIN, cameraId: c.id, off };
      }
    }
  }
  if (!best) return history;
  const { ts, cameraId } = best;
  return [...history, { ...frame, id: "lookalike", cameraId, ts, kind: "lookalike" as const }].sort((a, b) => a.ts - b.ts);
}
