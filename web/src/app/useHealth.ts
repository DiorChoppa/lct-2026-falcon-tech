import { useEffect, useState } from "react";
import { getHealth, getInfo } from "../api/client";
import type { Health, Info } from "../api/types";

const POLL_MS = 20_000;

export type HealthState = { kind: "loading" } | { kind: "ok" | "degraded"; health: Health } | { kind: "down" };

/** Готовность сервисов для индикатора в шапке: опрос раз в 20 с, пока вкладка видна. */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    let timer: number | undefined;
    let ctrl: AbortController | undefined;
    const tick = async () => {
      ctrl?.abort();
      ctrl = new AbortController();
      try {
        const health = await getHealth(ctrl.signal);
        setState({ kind: health.status === "ok" ? "ok" : "degraded", health });
      } catch (e) {
        if ((e as Error).name !== "AbortError") setState({ kind: "down" });
      }
      if (!document.hidden) timer = window.setTimeout(tick, POLL_MS);
    };
    const onVisibility = () => {
      window.clearTimeout(timer);
      if (!document.hidden) tick();
    };
    tick();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      ctrl?.abort();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return state;
}

/** Метаданные модели: имя, размерность и вход нужны экрану сверки деталей. */
export function useInfo(): Info | null {
  const [info, setInfo] = useState<Info | null>(null);
  useEffect(() => {
    getInfo()
      .then(setInfo)
      .catch(() => setInfo(null));
  }, []);
  return info;
}
