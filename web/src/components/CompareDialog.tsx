import { AlertTriangle, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { compare } from "../api/client";
import type { Bbox, Candidate, CompareResponse } from "../api/types";
import { pct } from "../lib/format";

// Сколько самых похожих пар областей рисовать: все взаимные совпадения (50–100 на
// пару) превращают кроп в сетку, оператору нужны самые сильные.
const TOP_MATCHES = 12;

type State =
  | { kind: "loading" }
  | { kind: "done"; data: CompareResponse }
  | { kind: "error"; message: string };

/**
 * Сверка по деталям: патч-токены запроса и кандидата из того же поиска
 * (GET /api/searches/{id}/compare/{galleryId}). Области приходят в координатах
 * входа модели (inputWidth × inputHeight); кроп растягивается в него целиком,
 * поэтому доли от входа — это доли от картинки.
 */
export function CompareDialog(props: {
  searchId: number;
  candidate: Candidate;
  rank: number;
  queryCropUrl: string;
  inputWidth: number;
  inputHeight: number;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [state, setState] = useState<State>({ kind: "loading" });
  const [active, setActive] = useState<number | null>(null);
  const galleryId = props.candidate.item.id;

  useEffect(() => {
    ref.current?.showModal();
  }, []);

  useEffect(() => {
    let alive = true;
    setState({ kind: "loading" });
    compare(props.searchId, galleryId)
      .then((data) => alive && setState({ kind: "done", data }))
      .catch((e) => alive && setState({ kind: "error", message: e instanceof Error ? e.message : String(e) }));
    return () => {
      alive = false;
    };
  }, [props.searchId, galleryId]);

  const top =
    state.kind === "done"
      ? [...state.data.matches].sort((a, b) => b.similarity - a.similarity).slice(0, TOP_MATCHES)
      : [];
  const patches = (props.inputWidth / 16) * (props.inputHeight / 16);

  return (
    <dialog
      ref={ref}
      className="dialog compare"
      aria-labelledby="compare-title"
      onClose={props.onClose}
      onClick={(e) => e.target === ref.current && ref.current?.close()}
    >
      <div className="dialog__head">
        <div>
          <p className="eyebrow">Сверка по деталям</p>
          <h2 id="compare-title">
            Запрос ↔ кандидат №{props.rank}
            <span className="compare__conf"> · уверенность {pct(props.candidate.confidence)}</span>
          </h2>
        </div>
        <button type="button" className="btn btn--ghost btn--icon" onClick={() => ref.current?.close()} aria-label="Закрыть">
          <X aria-hidden="true" />
        </button>
      </div>

      <div className="dialog__body" aria-busy={state.kind === "loading"}>
        <div className="compare__pair" onPointerLeave={() => setActive(null)}>
          <Crop
            title="Запрос"
            src={props.queryCropUrl}
            regions={top.map((m) => m.queryRegion)}
            w={props.inputWidth}
            h={props.inputHeight}
            active={active}
            onActive={setActive}
            loading={state.kind === "loading"}
          />
          <Crop
            title={`Кандидат · ${props.candidate.item.vehicleId ? `vehicle_id ${props.candidate.item.vehicleId}` : `id ${galleryId}`}`}
            src={props.candidate.item.cropUrl}
            regions={top.map((m) => m.candidateRegion)}
            w={props.inputWidth}
            h={props.inputHeight}
            active={active}
            onActive={setActive}
            loading={state.kind === "loading"}
          />
        </div>

        {state.kind === "loading" && (
          <p className="muted" role="status">
            Считаем патч-токены обоих кропов тем же ViT, что строит эмбеддинг…
          </p>
        )}
        {state.kind === "error" && (
          <div className="alert alert--error" role="alert">
            <AlertTriangle aria-hidden="true" />
            <span>{state.message}</span>
          </div>
        )}
        {state.kind === "done" && (
          <div className="compare__facts">
            <div className="compare__stat">
              <span className="compare__stat-value">{pct(state.data.localScore)}</span>
              <span className="compare__stat-label">счёт по деталям</span>
            </div>
            <div className="compare__stat">
              <span className="compare__stat-value">
                {state.data.matches.length}
                <span className="muted"> / {patches}</span>
              </span>
              <span className="compare__stat-label">взаимно совпавших патчей 16×16</span>
            </div>
            <p className="compare__explain">
              Цифрами отмечены {top.length} самых похожих пар: одинаковая цифра — одна и та же деталь на обоих
              снимках. Наведите на рамку, чтобы подсветить пару. Это не отдельная модель, а патч-токены того же
              графа, что считает эмбеддинг.
              {state.data.note === "few_shared_views" && (
                <strong> Ракурсы почти не пересекаются — общих деталей мало, решение держится на глобальном признаке.</strong>
              )}
            </p>
            {state.data.candidateTags.length > 0 && (
              <p className="compare__tags">
                Теги кандидата:{" "}
                {state.data.candidateTags.map((t) => `${t.key} ${pct(t.confidence)}`).join(", ")}
              </p>
            )}
          </div>
        )}
      </div>
    </dialog>
  );
}

function Crop(props: {
  title: string;
  src: string;
  regions: Bbox[];
  w: number;
  h: number;
  active: number | null;
  onActive: (i: number | null) => void;
  loading: boolean;
}) {
  const p = (v: number, total: number) => `${(v / total) * 100}%`;
  return (
    <figure className="compare__crop">
      <div className={`compare__frame${props.loading ? " is-scanning" : ""}${props.active != null ? " has-active" : ""}`}>
        <img src={props.src} alt={`${props.title}: кроп ТС`} />
        {props.regions.map((r, i) => (
          <span
            key={i}
            className={`compare__box${props.active === i ? " is-active" : ""}`}
            aria-hidden="true"
            onPointerEnter={() => props.onActive(i)}
            style={{ left: p(r.x, props.w), top: p(r.y, props.h), width: p(r.w, props.w), height: p(r.h, props.h) }}
          >
            <span className="compare__num">{i + 1}</span>
          </span>
        ))}
      </div>
      <figcaption>{props.title}</figcaption>
    </figure>
  );
}
