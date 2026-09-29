// Время маршрута — московское (UTC+3, без переходов на летнее время).
export const MIN = 60_000;
export const HOUR = 60 * MIN;
export const DAY = 24 * HOUR;
const MSK_OFFSET = 3 * HOUR;

/** Конец периода симуляции: 24.09.2026 00:00 по Москве. */
export const ANCHOR = Date.UTC(2026, 8, 23, 21, 0);
export const PERIOD_DAYS = 30;

export interface MskParts {
  /** YYYY-MM-DD по Москве. */
  dateKey: string;
  /** 0 — понедельник … 6 — воскресенье. */
  weekday: number;
  hour: number;
  minute: number;
}

export function msk(ts: number): MskParts {
  const d = new Date(ts + MSK_OFFSET);
  return {
    dateKey: d.toISOString().slice(0, 10),
    weekday: (d.getUTCDay() + 6) % 7,
    hour: d.getUTCHours(),
    minute: d.getUTCMinutes(),
  };
}

/** Полночь по Москве того же московского дня. */
export function mskMidnight(ts: number): number {
  return Math.floor((ts + MSK_OFFSET) / DAY) * DAY - MSK_OFFSET;
}

export function clock(ts: number): string {
  const { hour, minute } = msk(ts);
  return `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
}

const DAY_FMT = new Intl.DateTimeFormat("ru-RU", {
  weekday: "short",
  day: "numeric",
  month: "short",
  timeZone: "Europe/Moscow",
});

/** «чт, 24 сент.» */
export function dayLabel(ts: number): string {
  return DAY_FMT.format(ts);
}
