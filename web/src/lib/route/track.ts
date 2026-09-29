import { feasibility } from "./geo";
import type { Camera, Sighting } from "./types";

export interface Rejected {
  sighting: Sighting;
  /** Принятое появление, с которым несовместимо. */
  conflictWith: Sighting;
  km: number;
  gapMin: number;
  needKmh: number;
}

export interface Track {
  /** По времени. */
  accepted: Sighting[];
  rejected: Rejected[];
}

const RANK: Record<Sighting["kind"], number> = { real: 0, sim: 1, lookalike: 2 };

/**
 * Жадная сборка трека: сначала самые надёжные появления (реальные кадры,
 * затем смоделированные, затем похожие машины; внутри — по уверенности).
 * Каждое принимается, только если физически совместимо с ближайшими по
 * времени уже принятыми соседями; иначе — в отсеянное с причиной.
 */
export function buildTrack(sightings: Sighting[], cameras: Map<string, Camera>): Track {
  const order = [...sightings].sort(
    (a, b) => RANK[a.kind] - RANK[b.kind] || (b.confidence ?? 0) - (a.confidence ?? 0) || a.ts - b.ts,
  );
  const accepted: Sighting[] = [];
  const rejected: Rejected[] = [];
  for (const s of order) {
    if (!cameras.has(s.cameraId)) continue;
    let i = accepted.findIndex((x) => x.ts > s.ts);
    if (i < 0) i = accepted.length;
    const prev = accepted[i - 1];
    const next = accepted[i];
    const conflict = (prev && clash(prev, s, prev, cameras)) || (next && clash(s, next, next, cameras)) || null;
    if (conflict) rejected.push({ sighting: s, ...conflict });
    else accepted.splice(i, 0, s);
  }
  return { accepted, rejected };
}

function clash(a: Sighting, b: Sighting, other: Sighting, cameras: Map<string, Camera>): Omit<Rejected, "sighting"> | null {
  const f = feasibility(a, cameras.get(a.cameraId)!, b, cameras.get(b.cameraId)!);
  return f.ok ? null : { conflictWith: other, km: f.km, gapMin: f.gapMin, needKmh: f.needKmh };
}
