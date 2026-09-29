import { useEffect, useState } from "react";
import { search } from "../../api/client";
import type { Bbox } from "../../api/types";
import { isUsable } from "../../lib/bbox";
import { buildRoute, RouteError, type RouteData } from "../../lib/route/build";
import { SCENARIOS, scenarioFile } from "../search/scenarios";
import type { SearchSession } from "../search/useSearchSession";

export type RouteState =
  | { kind: "loading" }
  /** retry — сбой поиска или сети: повтор может помочь. */
  | { kind: "error"; message: string; retry: boolean }
  | { kind: "done"; data: RouteData; source: "session" | "scenario" };

const NO_QUERY = "Откройте маршрут из результатов поиска: нужен кадр, по которому модель найдёт эту машину.";

/**
 * Реальные кадры машины — повторный поиск с топ-100: сначала по запросу из
 * сессии поиска, если машины там нет или сессии нет — по сценарию демо с этой
 * машиной (так /route/959 со страницы решения работает всегда).
 */
export function useRoute(vehicleId: string | null, session: SearchSession): { state: RouteState; retry: () => void } {
  const [state, setState] = useState<RouteState>({ kind: "loading" });
  const [attempt, setAttempt] = useState(0);
  const { file, bbox, scenario: picked } = session;

  useEffect(() => {
    let alive = true;
    setState({ kind: "loading" });
    (async () => {
      if (!vehicleId) return setState({ kind: "error", message: "В адресе нет vehicle_id.", retry: false });
      // Кадр запроса — реальная точка маршрута, только если известно, что на нём эта же
      // машина (сценарий с тем же vehicle_id); иначе он чужой — берём лишь кадры галереи.
      const queries: {
        load: () => Promise<{ file: File; bbox: Bbox }>;
        source: "session" | "scenario";
        queryImageId: string | null;
      }[] = [];
      if (file && isUsable(bbox)) {
        const own = picked?.vehicleId === vehicleId ? picked.imageId : null;
        queries.push({ load: async () => ({ file, bbox }), source: "session", queryImageId: own });
      }
      const scenario = SCENARIOS.find((s) => s.vehicleId === vehicleId);
      if (scenario) {
        queries.push({
          load: async () => ({ file: await scenarioFile(scenario), bbox: scenario.bbox }),
          source: "scenario",
          queryImageId: scenario.imageId,
        });
      }

      let message = NO_QUERY;
      let retry = false;
      for (const q of queries) {
        try {
          const { file: f, bbox: b } = await q.load();
          const result = await search(f, b, 100);
          const data = buildRoute(vehicleId, result, q.queryImageId);
          if (alive) setState({ kind: "done", data, source: q.source });
          return;
        } catch (e) {
          message = e instanceof Error ? e.message : String(e);
          retry = !(e instanceof RouteError);
        }
      }
      if (alive) setState({ kind: "error", message, retry });
    })();
    return () => {
      alive = false;
    };
  }, [vehicleId, file, bbox, picked, attempt]);

  return { state, retry: () => setAttempt((n) => n + 1) };
}
