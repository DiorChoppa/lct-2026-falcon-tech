// Модель маршрута ТС (docs/superpowers/specs/2026-09-24-route-map-design.md §3).
// Всё пространственно-временное здесь смоделировано: в данных конкурса камер,
// времени и геопривязки нет (ТЗ §5).

export type CameraKind = "road" | "parking";
export type Zone = "residential" | "business" | "leisure";

export interface Camera {
  id: string;
  kind: CameraKind;
  lat: number;
  lon: number;
  address: string;
  city: string;
  /** Только у парковок Москвы: где генератор ищет ночную, дневную и досуговую стоянку. */
  zone?: Zone;
}

export interface TripRoute {
  id: string;
  name: string;
  /** Дорожные камеры по порядку от Москвы. */
  cameraIds: string[];
  /** Парковка в пункте назначения. */
  parkingId: string;
}

export interface Registry {
  cameras: Camera[];
  routes: TripRoute[];
}

export type SightingKind = "real" | "sim" | "lookalike";

export interface Sighting {
  id: string;
  cameraId: string;
  /** Начало появления, мс UTC; показывается по Москве. */
  ts: number;
  kind: SightingKind;
  /** Для sim — иллюстративный кроп этой же машины. */
  cropUrl: string;
  galleryId?: number;
  imageId?: string;
  vehicleId?: string | null;
  /** Уверенность ReID к запросу (real из поиска, lookalike). */
  confidence?: number;
  /** Кадр самого запроса оператора. */
  isQuery?: boolean;
  /** Длительность стоянки, минуты (парковки). */
  dwellMin?: number;
}

/** Реальный кадр до того, как генератор назначит ему время. */
export type RealFrame = Omit<Sighting, "id" | "ts" | "kind" | "dwellMin">;
