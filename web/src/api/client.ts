import type {
  Bbox,
  CompareResponse,
  GalleryItem,
  GalleryPage,
  Health,
  ImportReport,
  Info,
  SearchResponse,
} from "./types";
import * as mock from "./mock";

// ?mock=1 — интерфейс работает без бэкенда (разработка вёрстки, показ без стенда).
export const MOCK = new URLSearchParams(window.location.search).has("mock");

/** Лимит api-gateway на кадр (CreateForm/SearchForm в docs/openapi.json). */
export const MAX_IMAGE_BYTES = 3 * 1024 * 1024;
export const IMAGE_TYPES = ["image/jpeg", "image/png"];

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
  }
}

async function parse<T>(r: Response): Promise<T> {
  const body = await r.json().catch(() => null);
  if (!r.ok) {
    const message = body?.error?.message ?? `${r.status} ${r.statusText}`;
    throw new ApiError(message, r.status, body?.error?.code);
  }
  return body as T;
}

const bboxField = (b: Bbox) => JSON.stringify({ x: b.x, y: b.y, w: b.w, h: b.h });

export function getInfo(): Promise<Info> {
  if (MOCK) return mock.info();
  return fetch("/api/info").then((r) => parse<Info>(r));
}

/** 503 — тоже ответ: тело с флагами по сервисам, поэтому без parse(). */
export async function getHealth(signal?: AbortSignal): Promise<Health> {
  if (MOCK) return mock.health();
  const r = await fetch("/api/health", { signal, cache: "no-store" });
  const body = (await r.json().catch(() => null)) as Health | null;
  if (!body) throw new ApiError(`${r.status} ${r.statusText}`, r.status);
  return body;
}

export async function search(image: File, bbox: Bbox, topN = 10): Promise<SearchResponse> {
  if (MOCK) return mock.search(image, bbox);
  const form = new FormData();
  form.append("image", image);
  form.append("bbox", bboxField(bbox));
  form.append("top_n", String(topN));
  return fetch("/api/search", { method: "POST", body: form }).then((r) => parse<SearchResponse>(r));
}

export function compare(searchId: number, galleryId: number): Promise<CompareResponse> {
  if (MOCK) return mock.compare();
  return fetch(`/api/searches/${searchId}/compare/${galleryId}`).then((r) => parse<CompareResponse>(r));
}

export function exportUrl(searchId: number): string {
  return `/api/searches/${searchId}/export.csv`;
}

export function listGallery(page: number, pageSize: number, signal?: AbortSignal): Promise<GalleryPage> {
  if (MOCK) return mock.gallery(page, pageSize);
  const q = new URLSearchParams({ page: String(page), pageSize: String(pageSize) });
  return fetch(`/api/gallery?${q}`, { signal }).then((r) => parse<GalleryPage>(r));
}

export function addToGallery(input: {
  image: File;
  bbox: Bbox;
  vehicleId?: string;
  plate?: string;
}): Promise<GalleryItem> {
  if (MOCK) return mock.added(input.image, input.bbox);
  const form = new FormData();
  form.append("image", input.image);
  form.append("bbox", bboxField(input.bbox));
  form.append("image_id", input.image.name.replace(/\.[^.]+$/, ""));
  if (input.vehicleId) form.append("vehicle_id", input.vehicleId);
  if (input.plate) form.append("plate", input.plate);
  return fetch("/api/gallery", { method: "POST", body: form }).then((r) => parse<GalleryItem>(r));
}

export function setPlate(id: number, plate: string | null): Promise<GalleryItem> {
  if (MOCK) return mock.plate(id, plate);
  return fetch(`/api/gallery/${id}/plate`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plate }),
  }).then((r) => parse<GalleryItem>(r));
}

/** Импорт идёт синхронно, на тысячах кадров — минуты (nginx держит до 600 с). */
export function importGallery(csv: File, imagesDir: string): Promise<ImportReport> {
  if (MOCK) return mock.imported();
  const form = new FormData();
  form.append("csv", csv);
  if (imagesDir) form.append("images_dir", imagesDir);
  return fetch("/api/gallery/import", { method: "POST", body: form }).then((r) => parse<ImportReport>(r));
}

/** Проверка кадра до отправки: та же, что у api-gateway, но с понятным текстом. */
export function checkImage(file: File): string | null {
  if (!IMAGE_TYPES.includes(file.type)) return "Нужен кадр в JPEG или PNG.";
  if (file.size > MAX_IMAGE_BYTES)
    return `Кадр ${(file.size / 1024 / 1024).toFixed(1)} МиБ — больше лимита 3 МиБ. Пересохраните с меньшим качеством.`;
  return null;
}
