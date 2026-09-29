import type {
  Bbox,
  Candidate,
  CompareResponse,
  GalleryItem,
  GalleryPage,
  Health,
  ImportReport,
  Info,
  PatchMatch,
  SearchResponse,
} from "./types";

// Фикстуры для ?mock=1: экран работает без бэкенда. Картинки — кадры демо-сценариев.
const THUMBS = ["1-confident", "2-threshold", "3-refusal", "4-view"].map((n) => `/demo/${n}-thumb.webp`);
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

const item = (id: number, over: Partial<GalleryItem> = {}): GalleryItem => ({
  id,
  imageId: `mock${String(id).padStart(28, "0")}`,
  vehicleId: String(400 + (id % 37)),
  bbox: { x: 120, y: 80, w: 640, h: 480 },
  cropUrl: THUMBS[id % THUMBS.length],
  plate: id % 5 === 0 ? "А123ВС777" : null,
  tags: [],
  createdAt: new Date(Date.now() - id * 60_000).toISOString(),
  ...over,
});

export async function info(): Promise<Info> {
  await wait(150);
  return {
    service: "api-gateway (mock)",
    model: {
      name: "vitl16_dinov3_veriwild_tt_2plus2_fitv8",
      version: "1.2.0-final",
      dim: 1024,
      inputHeight: 256,
      inputWidth: 256,
      executionProvider: "cpu",
    },
    error: null,
  };
}

export async function health(): Promise<Health> {
  return { status: "ok", gallery: true, search: true, inference: true };
}

export async function search(image: File, bbox: Bbox): Promise<SearchResponse> {
  await wait(700);
  const url = URL.createObjectURL(image);
  const scores = [0.912, 0.874, 0.702, 0.641, 0.588, 0.553, 0.521, 0.498, 0.472, 0.455];
  const candidates: Candidate[] = scores.map((score, i) => ({
    item: item(100 + i, { vehicleId: i < 3 ? "959" : String(500 + i), bbox, cropUrl: i === 0 ? url : THUMBS[i % 4] }),
    score,
    confidence: score,
    localScore: null,
    accepted: score >= 0.661,
  }));
  return { id: 1, queryCropUrl: url, threshold: 0.6959864497184753, candidates };
}

export async function compare(): Promise<CompareResponse> {
  await wait(800);
  const matches: PatchMatch[] = Array.from({ length: 40 }, (_, i) => {
    const qx = (i * 37) % 240;
    const qy = (i * 53) % 240;
    return {
      queryRegion: { x: qx, y: qy, w: 16, h: 16 },
      candidateRegion: { x: (qx + 16) % 240, y: qy, w: 16, h: 16 },
      similarity: 0.95 - i * 0.01,
    };
  });
  return { matches, localScore: 0.41, note: "", candidateTags: [] };
}

export async function gallery(page: number, pageSize: number): Promise<GalleryPage> {
  await wait(300);
  const totalItems = 1500;
  const start = (page - 1) * pageSize;
  const data = Array.from({ length: Math.max(0, Math.min(pageSize, totalItems - start)) }, (_, i) =>
    item(totalItems - start - i),
  );
  return { data, pagination: { page, pageSize, totalItems, totalPages: Math.ceil(totalItems / pageSize) } };
}

export async function added(image: File, bbox: Bbox): Promise<GalleryItem> {
  await wait(600);
  return item(9999, { bbox, cropUrl: URL.createObjectURL(image), imageId: image.name });
}

export async function plate(id: number, plate: string | null): Promise<GalleryItem> {
  await wait(250);
  return item(id, { plate });
}

export async function imported(): Promise<ImportReport> {
  await wait(1200);
  return { imported: 748, failed: 2, errors: [{ imageId: "deadbeef", message: "кадр не найден" }] };
}
