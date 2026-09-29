import { plural } from "../format";
import { haversineKm } from "./geo";
import { DAY, dayLabel, HOUR, MIN, msk, mskMidnight } from "./time";
import type { Camera, Sighting } from "./types";

/** Стоянки ближе этого к центру места — то же место. */
export const PLACE_RADIUS_KM = 0.3;
/** С какого числа ночей / будних дней место считается регулярным. */
export const MIN_REGULAR = 4;
const NIGHT_HOUR = 3; // ночь «накрыта», если стоянка содержит 03:00 по Москве
const DAY_HOUR = 13; // будний день «накрыт», если стоянка содержит 13:00 пн–пт

export interface Place {
  id: string;
  lat: number;
  lon: number;
  city: string;
  address: string;
  stays: Sighting[];
  hours: number;
  /** Разных дат со стоянкой здесь. */
  days: number;
  /** Даты утра ночей, проведённых здесь. */
  nights: string[];
  /** Будние даты, днём проведённые здесь. */
  weekdays: string[];
}

export type PlaceRole = "night" | "day" | "trip" | "episodic";

export interface PlaceHypothesis {
  place: Place;
  role: PlaceRole;
  /** Только у регулярных мест. */
  confidence: number | null;
  /** Факт: «Регулярная ночная стоянка». */
  fact: string;
  /** Гипотеза для оператора: «вероятно, место жительства». */
  guess: string | null;
  /** Почему: «14 из 17 наблюдённых ночей, в среднем 22:40–07:10». */
  explain: string;
}

/** Даты, в которые стоянка содержит момент hourMsk по Москве. */
export function coveredDates(s: Sighting, hourMsk: number, weekdaysOnly: boolean): string[] {
  if (!s.dwellMin) return [];
  const end = s.ts + s.dwellMin * MIN;
  const dates: string[] = [];
  for (let day = mskMidnight(s.ts); day <= end; day += DAY) {
    const t = day + hourMsk * HOUR;
    if (t < s.ts || t > end) continue;
    const p = msk(t);
    if (weekdaysOnly && p.weekday > 4) continue;
    dates.push(p.dateKey);
  }
  return dates;
}

/** Жадная кластеризация стоянок: событий мало, DBSCAN не нужен. По убыванию часов. */
export function clusterPlaces(accepted: Sighting[], cameras: Map<string, Camera>): Place[] {
  const places: Place[] = [];
  for (const s of accepted) {
    const cam = cameras.get(s.cameraId);
    if (!s.dwellMin || !cam) continue;
    let p = places.find((x) => haversineKm(x, cam) <= PLACE_RADIUS_KM);
    if (!p) {
      p = { id: `place-${places.length + 1}`, lat: cam.lat, lon: cam.lon, city: cam.city, address: cam.address, stays: [], hours: 0, days: 0, nights: [], weekdays: [] };
      places.push(p);
    }
    const k = p.stays.length;
    p.lat = (p.lat * k + cam.lat) / (k + 1);
    p.lon = (p.lon * k + cam.lon) / (k + 1);
    p.stays.push(s);
    p.hours += s.dwellMin / 60;
  }
  for (const p of places) {
    p.nights = [...new Set(p.stays.flatMap((s) => coveredDates(s, NIGHT_HOUR, false)))];
    p.weekdays = [...new Set(p.stays.flatMap((s) => coveredDates(s, DAY_HOUR, true)))];
    p.days = new Set(p.stays.map((s) => msk(s.ts).dateKey)).size;
  }
  return places.sort((a, b) => b.hours - a.hours);
}

/** Доля × поправка на малую выборку, не выше 0,95: это гипотеза, а не факт. */
export function confidence(hits: number, observed: number): number {
  if (!observed) return 0;
  return Math.min(0.95, (hits / observed) * (1 - Math.exp(-hits / 5)));
}

/** Среднее время суток по кругу (23:00 и 01:00 → 00:00). */
export function meanClock(minutes: number[]): string {
  const a = minutes.map((m) => (m / 1440) * 2 * Math.PI);
  const ang = Math.atan2(a.reduce((s, x) => s + Math.sin(x), 0), a.reduce((s, x) => s + Math.cos(x), 0));
  const m = Math.round((((ang / (2 * Math.PI)) * 1440) % 1440 + 1440) % 1440);
  return `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

const minuteOfDay = (ts: number) => {
  const p = msk(ts);
  return p.hour * 60 + p.minute;
};

function stayWindow(stays: Sighting[]): string {
  const from = meanClock(stays.map((s) => minuteOfDay(s.ts)));
  const to = meanClock(stays.map((s) => minuteOfDay(s.ts + (s.dwellMin ?? 0) * MIN)));
  return `${from}–${to}`;
}

const ORDER: Record<PlaceRole, number> = { night: 0, day: 1, trip: 2, episodic: 3 };

export function hypotheses(places: Place[]): PlaceHypothesis[] {
  const nightsObserved = new Set(places.flatMap((p) => p.nights)).size;
  const weekdaysObserved = new Set(places.flatMap((p) => p.weekdays)).size;
  const out = places.map((place): PlaceHypothesis => {
    if (place.nights.length >= MIN_REGULAR) {
      const stays = place.stays.filter((s) => coveredDates(s, NIGHT_HOUR, false).length);
      return {
        place,
        role: "night",
        confidence: confidence(place.nights.length, nightsObserved),
        fact: "Регулярная ночная стоянка",
        guess: "вероятно, место жительства",
        explain: `${place.nights.length} из ${nightsObserved} наблюдённых ночей, в среднем ${stayWindow(stays)}`,
      };
    }
    if (place.weekdays.length >= MIN_REGULAR) {
      const stays = place.stays.filter((s) => coveredDates(s, DAY_HOUR, true).length);
      return {
        place,
        role: "day",
        confidence: confidence(place.weekdays.length, weekdaysObserved),
        fact: "Регулярная дневная стоянка по будням",
        guess: "вероятно, место работы или учёбы",
        explain: `${place.weekdays.length} из ${weekdaysObserved} наблюдённых будних дней, в среднем ${stayWindow(stays)}`,
      };
    }
    const visits = `${place.stays.length} ${plural(place.stays.length, ["визит", "визита", "визитов"])}, ${Math.round(place.hours)} ч`;
    if (place.city !== "Москва") {
      const from = dayLabel(place.stays[0].ts);
      const last = place.stays[place.stays.length - 1];
      const to = dayLabel(last.ts + (last.dwellMin ?? 0) * MIN);
      const dates = from === to ? from : `${from} — ${to}`;
      return { place, role: "trip", confidence: null, fact: `Поездка: ${place.city}`, guess: null, explain: `${dates} · ${visits}` };
    }
    return { place, role: "episodic", confidence: null, fact: "Эпизодическое место", guess: null, explain: visits };
  });
  return out.sort((a, b) => ORDER[a.role] - ORDER[b.role] || b.place.hours - a.place.hours);
}
