import { CheckCircle2, Download, Route as RouteIcon, ScanEye, ShieldQuestion, Timer } from "lucide-react";
import { useState, type CSSProperties } from "react";
import { exportUrl } from "../api/client";
import { Link, routeHref } from "../app/router";
import type { Candidate, SearchResponse } from "../api/types";
import { dec, duration, pct, plural } from "../lib/format";
import { CompareDialog } from "./CompareDialog";

const CANDIDATE_FORMS: [string, string, string] = ["кандидат", "кандидата", "кандидатов"];

/**
 * Результат поиска: вердикт (кандидаты или отказ), распределение уверенности
 * относительно порога, карточки топ-N. При отказе топ-N всё равно показан —
 * оператор может проверить глазами, но система за совпадение не ручается.
 */
export function Results(props: {
  result: SearchResponse;
  ms: number;
  stale: boolean;
  onRerun: () => void;
  inputSize: { w: number; h: number };
}) {
  const { result } = props;
  const [comparing, setComparing] = useState<number | null>(null);
  const accepted = result.candidates.filter((c) => c.accepted);
  const refused = accepted.length === 0;
  const best = result.candidates[0];

  if (result.candidates.length === 0) {
    return (
      <div className="empty card">
        <ShieldQuestion aria-hidden="true" />
        <h3>Галерея пуста</h3>
        <p className="muted">Добавьте ТС на экране «Галерея» или импортом CSV — искать пока не среди чего.</p>
      </div>
    );
  }

  return (
    <div className={`results${props.stale ? " is-stale" : ""}`}>
      <section className={`verdict card ${refused ? "verdict--refused" : "verdict--ok"}`} aria-labelledby="verdict-title">
        <div className="verdict__query reticle">
          <img src={result.queryCropUrl} alt="Кроп запроса" />
          <span className="verdict__query-label">запрос</span>
        </div>
        <div className="verdict__body">
          <div className="verdict__status">
            {refused ? <ShieldQuestion aria-hidden="true" /> : <CheckCircle2 aria-hidden="true" />}
            <span>{refused ? "Отказ от идентификации" : "Совпадение найдено"}</span>
          </div>
          <h2 id="verdict-title" className="verdict__title">
            {refused ? (
              <>Уверенного совпадения в галерее нет</>
            ) : (
              <>
                {accepted.length} {plural(accepted.length, CANDIDATE_FORMS)} выше порога
              </>
            )}
          </h2>
          <p className="verdict__text">
            {refused ? (
              <>
                Лучшая уверенность {pct(best.confidence)} ниже порога {dec(result.threshold)}. Ближайшие{" "}
                {result.candidates.length} показаны ниже только для ручной проверки.
              </>
            ) : (
              <>
                Лучший кандидат — {pct(best.confidence)} при пороге {dec(result.threshold)}. Кандидаты ниже порога
                оставлены в списке с пометкой.
              </>
            )}
          </p>
          <div className="verdict__meta">
            <span className="badge">
              <Timer aria-hidden="true" />
              {duration(props.ms)}
            </span>
            <span className="badge mono">поиск #{result.id}</span>
            <a className="btn btn--sm" href={exportUrl(result.id)} download>
              <Download aria-hidden="true" />
              Экспорт CSV
            </a>
          </div>
        </div>
      </section>

      {props.stale && (
        <div className="stale-bar" role="status">
          <span>Кадр или рамка изменились — результаты относятся к предыдущему запросу.</span>
          <button type="button" className="btn btn--primary btn--sm" onClick={props.onRerun}>
            Повторить поиск
          </button>
        </div>
      )}

      <ConfidenceChart candidates={result.candidates} threshold={result.threshold} />

      <ol className="cands" aria-label="Кандидаты по убыванию уверенности">
        {result.candidates.map((c, i) => (
          <CandidateCard key={c.item.id} c={c} rank={i + 1} threshold={result.threshold} onCompare={() => setComparing(i)} />
        ))}
      </ol>

      {comparing != null && (
        <CompareDialog
          searchId={result.id}
          candidate={result.candidates[comparing]}
          rank={comparing + 1}
          queryCropUrl={result.queryCropUrl}
          inputWidth={props.inputSize.w}
          inputHeight={props.inputSize.h}
          onClose={() => setComparing(null)}
        />
      )}
    </div>
  );
}

/**
 * Уверенность топ-N столбиками на шкале 0–1 с линией порога. Это и есть режим
 * отказа в одной картинке: столбик выше линии — принят. Значения дублируются в
 * карточках ниже (табличный вид), поэтому подсказка только дополняет.
 */
function ConfidenceChart({ candidates, threshold }: { candidates: Candidate[]; threshold: number }) {
  return (
    <figure className="conf-chart card">
      <figcaption className="conf-chart__head">
        <span className="conf-chart__title">Уверенность топ-{candidates.length}</span>
        <span className="conf-chart__legend">
          <span className="legend-key legend-key--ok" aria-hidden="true" /> принят
          <span className="legend-key legend-key--below" aria-hidden="true" /> ниже порога
          <span className="legend-key legend-key--line" aria-hidden="true" /> порог {dec(threshold)}
        </span>
      </figcaption>
      <div className="conf-chart__plot">
        <div className="conf-chart__threshold" style={{ bottom: `${threshold * 100}%` }} aria-hidden="true" />
        {candidates.map((c, i) => (
          <a
            key={c.item.id}
            href={`#cand-${i + 1}`}
            className={`conf-chart__bar${c.accepted ? " is-ok" : ""}`}
            style={{ "--h": `${Math.max(c.confidence, 0) * 100}%`, "--i": i } as CSSProperties}
            aria-label={`Кандидат ${i + 1}: ${pct(c.confidence)}, ${c.accepted ? "принят" : "ниже порога"}`}
            data-tip={`№${i + 1} · ${pct(c.confidence, 1)}`}
          >
            {i === 0 && <span className="conf-chart__value">{pct(c.confidence)}</span>}
          </a>
        ))}
      </div>
      <div className="conf-chart__axis" aria-hidden="true">
        {candidates.map((c, i) => (
          <span key={c.item.id}>{i + 1}</span>
        ))}
      </div>
    </figure>
  );
}

function CandidateCard(props: { c: Candidate; rank: number; threshold: number; onCompare: () => void }) {
  const { c, rank } = props;
  return (
    <li id={`cand-${rank}`} className={`cand card${c.accepted ? " cand--ok" : ""}`} style={{ "--i": rank } as CSSProperties}>
      <div className="cand__media reticle">
        <img src={c.item.cropUrl} alt={`Кандидат ${rank}${c.item.vehicleId ? `, vehicle_id ${c.item.vehicleId}` : ""}`} loading="lazy" />
        <span className="cand__rank">№{rank}</span>
      </div>
      <div className="cand__body">
        <div className="cand__score">
          <span className="cand__pct">{pct(c.confidence)}</span>
          <span className={c.accepted ? "badge badge--ok" : "badge"}>
            {c.accepted ? <CheckCircle2 aria-hidden="true" /> : null}
            {c.accepted ? "принят" : "ниже порога"}
          </span>
        </div>
        <div className="meter" aria-hidden="true">
          <div className={`meter__fill${c.accepted ? " meter__fill--ok" : ""}`} style={{ width: `${c.confidence * 100}%` }} />
          <div className="meter__tick" style={{ left: `${props.threshold * 100}%` }} />
        </div>
        <dl className="cand__meta">
          <div>
            <dt>ГРЗ</dt>
            <dd>{c.item.plate ? <span className="plate">{c.item.plate}</span> : <span className="muted">не привязан</span>}</dd>
          </div>
          <div>
            <dt>vehicle_id</dt>
            <dd className="mono">{c.item.vehicleId ?? <span className="muted">—</span>}</dd>
          </div>
          <div>
            <dt>кадр</dt>
            <dd className="mono cand__image" title={c.item.imageId}>
              {c.item.imageId}
            </dd>
          </div>
          {c.localScore != null && (
            <div>
              <dt>детали</dt>
              <dd>{pct(c.localScore)}</dd>
            </div>
          )}
        </dl>
        <div className="cand__actions">
          <button type="button" className="btn btn--sm" onClick={props.onCompare}>
            <ScanEye aria-hidden="true" />
            Сверить детали
          </button>
          {c.item.vehicleId ? (
            <Link href={routeHref(c.item.vehicleId)} className="btn btn--sm">
              <RouteIcon aria-hidden="true" />
              Маршрут
            </Link>
          ) : (
            <>
              <button type="button" className="btn btn--sm" disabled aria-describedby={`no-route-${c.item.id}`}>
                <RouteIcon aria-hidden="true" />
                Маршрут
              </button>
              <p id={`no-route-${c.item.id}`} className="cand__hint">
                Нет vehicle_id — идентичность не подтверждена
              </p>
            </>
          )}
        </div>
      </div>
    </li>
  );
}
