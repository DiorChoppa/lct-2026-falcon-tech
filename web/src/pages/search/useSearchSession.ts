import { useCallback, useEffect, useRef, useState } from "react";
import { checkImage, search } from "../../api/client";
import type { Bbox, SearchResponse } from "../../api/types";
import { isUsable } from "../../lib/bbox";
import { scenarioFile, type Scenario } from "./scenarios";

export type SearchState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: SearchResponse; ms: number; stale: boolean }
  | { kind: "error"; message: string };

export interface SearchSession {
  file: File | null;
  fileUrl: string | null;
  fileError: string | null;
  bbox: Bbox | null;
  scenario: Scenario | null;
  state: SearchState;
  pickFile: (f: File | null) => void;
  setBbox: (b: Bbox | null) => void;
  loadScenario: (s: Scenario) => Promise<void>;
  run: () => Promise<void>;
  canRun: boolean;
}

/** Кадр, bbox и результат поиска; живёт в App, чтобы переживать смену экранов. */
export function useSearchSession(): SearchSession {
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [bbox, setBboxState] = useState<Bbox | null>(null);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [state, setState] = useState<SearchState>({ kind: "idle" });
  const runId = useRef(0);

  useEffect(() => {
    if (!file) {
      setFileUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Кадр или рамка поменялись после поиска — кандидаты уже не про этот запрос.
  const markStale = () => setState((s) => (s.kind === "done" ? { ...s, stale: true } : s));

  const pickFile = useCallback((f: File | null) => {
    const err = f ? checkImage(f) : null;
    setFileError(err);
    if (err) return;
    setFile(f);
    setBboxState(null);
    setScenario(null);
    markStale();
  }, []);

  const setBbox = useCallback((b: Bbox | null) => {
    setBboxState(b);
    markStale();
  }, []);

  const loadScenario = useCallback(async (s: Scenario) => {
    setFileError(null);
    try {
      setFile(await scenarioFile(s));
      setBboxState(s.bbox);
      setScenario(s);
      setState({ kind: "idle" });
    } catch {
      setFileError("Не удалось загрузить кадр сценария.");
    }
  }, []);

  const canRun = !!file && isUsable(bbox) && state.kind !== "loading";

  const run = useCallback(async () => {
    if (!file || !isUsable(bbox)) return;
    const id = ++runId.current;
    setState({ kind: "loading" });
    const t0 = performance.now();
    try {
      const result = await search(file, bbox);
      if (id === runId.current) setState({ kind: "done", result, ms: performance.now() - t0, stale: false });
    } catch (e) {
      if (id === runId.current) setState({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, [file, bbox]);

  return { file, fileUrl, fileError, bbox, scenario, state, pickFile, setBbox, loadScenario, run, canRun };
}
