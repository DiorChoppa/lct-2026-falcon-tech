// Типы по docs/openapi.json; менять только вместе с бэкендом.

export interface Bbox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface GalleryItem {
  id: number;
  imageId: string;
  vehicleId: string | null;
  bbox: Bbox;
  cropUrl: string;
  plate: string | null;
  tags: unknown[];
  createdAt: string;
}

export interface Candidate {
  item: GalleryItem;
  /** Косинусное сходство глобальных эмбеддингов, 0..1. */
  score: number;
  /** Уверенность, к которой применяется порог (alpha*score + (1-alpha)*localScore). */
  confidence: number;
  /** Счёт по деталям, если считался. */
  localScore: number | null;
  /** Прошёл порог режима отказа. */
  accepted: boolean;
}

export interface SearchResponse {
  /** id поиска — для экспорта /api/searches/{id}/export.csv. */
  id: number;
  queryCropUrl: string;
  threshold: number;
  /** Топ-N по убыванию уверенности, включая ниже порога. Ни одного accepted = отказ. */
  candidates: Candidate[];
}

export interface PatchMatch {
  /** Область в координатах входа модели (inputWidth × inputHeight из /api/info). */
  queryRegion: Bbox;
  candidateRegion: Bbox;
  similarity: number;
}

export interface Tag {
  key: string;
  confidence: number;
  region: Bbox | null;
}

/** GET /api/searches/{id}/compare/{galleryId}: сверка пары по патч-токенам. */
export interface CompareResponse {
  matches: PatchMatch[];
  /** Доля взаимно совпавших патчей с весом по сходству, 0..1. */
  localScore: number;
  /** "few_shared_views", если ракурсы почти не пересекаются. */
  note: string;
  candidateTags: Tag[];
}

export interface ModelInfo {
  name: string;
  version: string;
  dim: number;
  inputHeight: number;
  inputWidth: number;
  executionProvider: string;
}

export interface Info {
  service: string;
  model: ModelInfo | null;
  error: string | null;
}

export interface ApiError {
  error: { code: string; message: string };
}

/** GET /api/health: "ok" или "degraded", если хоть один gRPC-сервис не SERVING. */
export interface Health {
  status: string;
  gallery: boolean;
  search: boolean;
  inference: boolean;
}

export interface Pagination {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

/** GET /api/gallery: новые записи первыми. */
export interface GalleryPage {
  data: GalleryItem[];
  pagination: Pagination;
}

export interface ImportError {
  imageId: string;
  message: string;
}

/** POST /api/gallery/import */
export interface ImportReport {
  imported: number;
  failed: number;
  errors: ImportError[];
}
