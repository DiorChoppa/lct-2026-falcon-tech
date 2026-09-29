import type { Bbox } from "../../api/types";

/**
 * Сценарии демо из docs/demo.md: запросы development-эпизода SEARCH, их пары
 * есть в галерее стенда с vehicle_id, поэтому верность совпадения видна в карточке.
 * Кадры — web/public/demo (копии dataset/images/<imageId>.jpg).
 */
export interface Scenario {
  id: string;
  title: string;
  expect: string;
  imageId: string;
  frame: string;
  thumb: string;
  bbox: Bbox;
  /** vehicle_id машины запроса — для прямой ссылки на её маршрут. */
  vehicleId?: string;
}

const demo = (name: string) => ({ frame: `/demo/${name}.jpg`, thumb: `/demo/${name}-thumb.webp` });

export const SCENARIOS: Scenario[] = [
  {
    id: "confident",
    vehicleId: "959",
    title: "Уверенное совпадение",
    expect: "4 принятых кандидата, все vehicle_id\u00a0959, топ-1 ≈\u00a091\u00a0%",
    imageId: "6774d18f78dd4198a14977ec60ec0c3c",
    bbox: { x: 603, y: 154, w: 952, h: 925 },
    ...demo("1-confident"),
  },
  {
    id: "threshold",
    vehicleId: "1115",
    title: "Трудный случай у порога",
    expect: "принят 1 кандидат, vehicle_id\u00a01115, ≈\u00a00,67 при пороге 0,66",
    imageId: "ccb016f8bf3145f38c39974370b7b68b",
    bbox: { x: 214, y: 173, w: 562, h: 392 },
    ...demo("2-threshold"),
  },
  {
    id: "refusal",
    title: "Честный отказ",
    expect: "машины нет в галерее — отказ, топ-N помечен «ниже порога»",
    imageId: "5e9a3988152c4aa49e0d5e8191201bca",
    bbox: { x: 659, y: 76, w: 1016, h: 632 },
    ...demo("3-refusal"),
  },
  {
    id: "view",
    vehicleId: "1420",
    title: "Ошибка: другой ракурс",
    expect: "отказ: топ-3 — чужой белый Polo, тоже сзади; верный 1420 (спереди) — 4-й, ≈\u00a042\u00a0%",
    imageId: "17235c85ac6e46dfaee5cadec0b0567e",
    bbox: { x: 573, y: 297, w: 680, h: 411 },
    ...demo("4-view"),
  },
];

/** Кадр сценария как File — тот же путь, что у загрузки своего кадра. */
export async function scenarioFile(s: Scenario): Promise<File> {
  const r = await fetch(s.frame);
  if (!r.ok) throw new Error(`Кадр сценария не загрузился: ${r.status}`);
  return new File([await r.blob()], `${s.imageId}.jpg`, { type: "image/jpeg" });
}
