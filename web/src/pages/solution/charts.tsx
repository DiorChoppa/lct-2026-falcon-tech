import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import data from "../../data/threshold.json";
import { dec, int, pct } from "../../lib/format";

// Графики страницы «Решение». Палитра — первые три категориальных слота
// dataviz-скилла (проверены валидатором на наших поверхностях, charts.css);
// у каждого графика легенда, прямые подписи и табличный вид в <details>.

type Point = (typeof data.curve)[number];
type Group = "correct" | "wrong" | "absent";

const SERIES = [
  { key: "f1", label: "F1", cls: "s1" },
  { key: "tnr", label: "TNR", cls: "s2" },
  { key: "score", label: "Балл 0,7·F1 + 0,3·TNR", cls: "s3" },
] as const;

const GROUPS: { key: Group; label: string; cls: string }[] = [
  { key: "correct", label: "пара есть, топ-1 верный", cls: "s1" },
  { key: "wrong", label: "пара есть, топ-1 чужой", cls: "s2" },
  { key: "absent", label: "пары в галерее нет", cls: "s3" },
];

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

const X0 = 0.3;
const X1 = 0.95;
const Y0 = 0;

/**
 * Кривая F1 / TNR / балла жюри от порога на эпизоде калибровки (косинус, как в живом сервисе).
 * Перекрестье ищет ближайший порог; стрелками ←/→ — то же с клавиатуры.
 */
export function ThresholdChart() {
  const [box, width] = useWidth<HTMLDivElement>();
  const height = 320;
  const m = { top: 20, right: width < 520 ? 16 : 150, bottom: 40, left: 44 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const x = (t: number) => m.left + ((t - X0) / (X1 - X0)) * iw;
  const y = (v: number) => m.top + (1 - (Math.max(v, Y0) - Y0) / (1 - Y0)) * ih;
  // Ближайшая к порогу точка сетки — старт для навигации стрелками.
  const opIndex = useMemo(() => Math.round((data.threshold - X0) / 0.005), []);
  const [hover, setHover] = useState<number | null>(null);
  // Без наведения — ровно рабочая точка, а не соседний узел сетки.
  const active: Point = hover == null ? data.operatingPoint : data.curve[hover];

  const path = (k: "f1" | "tnr" | "score") =>
    data.curve.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = X0 + ((e.clientX - r.left - m.left) / iw) * (X1 - X0);
    const i = Math.round((t - X0) / 0.005);
    setHover(Math.min(Math.max(i, 0), data.curve.length - 1));
  };
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const cur = hover ?? opIndex;
    setHover(Math.min(Math.max(cur + (e.key === "ArrowRight" ? 1 : -1), 0), data.curve.length - 1));
  };

  const last = data.curve[data.curve.length - 1];
  const xTicks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const tipLeft = x(active.t) > width - 200;

  return (
    <figure className="chart card">
      <figcaption className="chart__head">
        <span className="chart__title">Качество режима отказа в зависимости от порога</span>
        <span className="chart__sub">
          эпизод калибровки · {data.queries.length} запросов, из них {data.operatingPoint.tn + data.operatingPoint.fp} без пары
        </span>
        <Legend items={SERIES.map((s) => ({ label: s.label, cls: s.cls, line: true }))} />
      </figcaption>
      <div ref={box} className="chart__box">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Кривая F1, TNR и балла жюри от порога. Максимум балла ${dec(data.operatingPoint.score)} при пороге ${dec(data.threshold)}. Стрелками влево и вправо можно двигать перекрестье.`}
          tabIndex={0}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
        >
          {yTicks.map((v) => (
            <g key={v}>
              <line className="chart__grid" x1={m.left} x2={m.left + iw} y1={y(v)} y2={y(v)} />
              <text className="chart__tick" x={m.left - 8} y={y(v)} dy="0.32em" textAnchor="end">
                {dec(v, 1)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} className="chart__tick" x={x(t)} y={m.top + ih + 18} textAnchor="middle">
              {dec(t, 1)}
            </text>
          ))}
          <text className="chart__axis-label" x={m.left + iw} y={height - 4} textAnchor="end">
            порог уверенности топ-1 →
          </text>
          <line className="chart__marker" x1={x(data.threshold)} x2={x(data.threshold)} y1={m.top} y2={m.top + ih} />
          <text className="chart__marker-label" x={x(data.threshold) + 6} y={m.top + ih - 8}>
            порог {dec(data.threshold)}
          </text>

          {SERIES.map((s) => (
            <path key={s.key} className={`chart__line ${s.cls}`} d={path(s.key)} />
          ))}
          {m.right > 100 &&
            endLabels(SERIES.map((s) => ({ s, y: y(last[s.key]) }))).map(({ s, y: ly }) => (
              <text key={s.key} className="chart__end-label" x={x(last.t) + 8} y={ly} dy="0.32em">
                {s.key === "score" ? "балл" : s.label} {dec(last[s.key], 2)}
              </text>
            ))}

          {hover != null && <line className="chart__crosshair" x1={x(active.t)} x2={x(active.t)} y1={m.top} y2={m.top + ih} />}
          {SERIES.map((s) => (
            <circle key={s.key} className={`chart__dot ${s.cls}`} cx={x(active.t)} cy={y(active[s.key])} r={4.5} />
          ))}
        </svg>
        {hover != null && <div
          className="chart__tip"
          style={{
            left: tipLeft ? undefined : x(active.t) + 12,
            right: tipLeft ? width - x(active.t) + 12 : undefined,
            top: m.top,
          }}
          aria-live="polite"
        >
          <div className="chart__tip-head">порог {dec(active.t)}</div>
          {SERIES.map((s) => (
            <div key={s.key} className="chart__tip-row">
              <span className={`key-line ${s.cls}`} aria-hidden="true" />
              <strong>{dec(active[s.key])}</strong>
              <span>{s.label}</span>
            </div>
          ))}
          <div className="chart__tip-cm mono">
            TP {active.tp} · FP {active.fp} · FN {active.fn} · TN {active.tn}
          </div>
        </div>}
      </div>
      <TableView caption="Точки кривой (шаг 0,01)">
        <thead>
          <tr>
            <th>порог</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>TN</th>
            <th>F1</th>
            <th>TNR</th>
            <th>балл</th>
          </tr>
        </thead>
        <tbody>
          {data.curve
            .filter((_, i) => i % 2 === 0)
            .map((p) => (
              <tr key={p.t}>
                <td>{dec(p.t, 2)}</td>
                <td>{p.tp}</td>
                <td>{p.fp}</td>
                <td>{p.fn}</td>
                <td>{p.tn}</td>
                <td>{dec(p.f1)}</td>
                <td>{dec(p.tnr)}</td>
                <td>{dec(p.score)}</td>
              </tr>
            ))}
        </tbody>
      </TableView>
    </figure>
  );
}

/** Подписи концов линий: сверху вниз, не ближе 15 px друг к другу. */
function endLabels<T extends { y: number }>(items: T[]): T[] {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  sorted.forEach((it, i) => {
    if (i > 0 && it.y - sorted[i - 1].y < 15) it.y = sorted[i - 1].y + 15;
  });
  return sorted;
}

/** Уверенность топ-1 каждого запроса, по строке на группу; линия — порог. */
export function ConfidenceStrip() {
  const [box, width] = useWidth<HTMLDivElement>();
  const rowH = 56;
  const m = { top: 30, right: 20, bottom: 36, left: width < 560 ? 12 : 196 };
  const iw = width - m.left - m.right;
  const height = m.top + rowH * GROUPS.length + m.bottom;
  const x = (c: number) => m.left + ((c - 0.15) / (1 - 0.15)) * iw;
  const [hover, setHover] = useState<{ g: number; i: number } | null>(null);

  // Детерминированный разброс по вертикали, чтобы точки не сливались.
  const rows = useMemo(
    () =>
      GROUPS.map((g) => {
        const pts = data.queries.filter((q) => q.g === g.key).map((q) => q.c);
        return { ...g, pts, above: pts.filter((c) => c >= data.threshold).length };
      }),
    [],
  );
  const jitter = (i: number) => (((i * 2654435761) % 1000) / 1000 - 0.5) * (rowH - 26);

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    const g = Math.floor((e.clientY - r.top - m.top) / rowH);
    if (g < 0 || g >= rows.length) return setHover(null);
    let best = -1;
    let bd = 14;
    rows[g].pts.forEach((c, i) => {
      const d = Math.abs(x(c) - px);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    setHover(best >= 0 ? { g, i: best } : null);
  };

  const hv = hover ? rows[hover.g].pts[hover.i] : null;

  return (
    <figure className="chart card">
      <figcaption className="chart__head">
        <span className="chart__title">Уверенность топ-1 по каждому запросу</span>
        <span className="chart__sub">Одна точка — один запрос; справа от линии сервис отвечает, слева отказывает</span>
      </figcaption>
      <div ref={box} className="chart__box">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Точечная диаграмма уверенности. ${rows
            .map((r) => `${r.label}: ${r.pts.length} запросов, выше порога ${r.above}`)
            .join("; ")}.`}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {[0.2, 0.4, 0.6, 0.8, 1].map((t) => (
            <g key={t}>
              <line className="chart__grid" x1={x(t)} x2={x(t)} y1={m.top} y2={m.top + rowH * rows.length} />
              <text className="chart__tick" x={x(t)} y={height - 14} textAnchor="middle">
                {dec(t, 1)}
              </text>
            </g>
          ))}
          <rect
            className="chart__reject"
            x={m.left}
            y={m.top}
            width={x(data.threshold) - m.left}
            height={rowH * rows.length}
          />
          {rows.map((r, g) => {
            const cy = m.top + rowH * g + rowH / 2;
            return (
              <g key={r.key}>
                {m.left > 100 && (
                  <>
                    <text className="chart__row-label" x={m.left - 14} y={cy - 7} textAnchor="end">
                      {r.label}
                    </text>
                    <text className="chart__row-sub" x={m.left - 14} y={cy + 11} textAnchor="end">
                      {r.pts.length} · принято {r.above}
                    </text>
                  </>
                )}
                {r.pts.map((c, i) => (
                  <circle
                    key={i}
                    className={`chart__pt ${r.cls}${hover && hover.g === g && hover.i === i ? " is-active" : ""}`}
                    cx={x(c)}
                    cy={cy + jitter(i + g * 97)}
                    r={4}
                  />
                ))}
              </g>
            );
          })}
          <line
            className="chart__marker"
            x1={x(data.threshold)}
            x2={x(data.threshold)}
            y1={m.top - 4}
            y2={m.top + rowH * rows.length}
          />
          <text className="chart__marker-label" x={x(data.threshold) - 6} y={m.top - 12} textAnchor="end">
            ← отказ
          </text>
          <text className="chart__marker-label" x={x(data.threshold) + 6} y={m.top - 12}>
            ответ → · порог {dec(data.threshold)}
          </text>
        </svg>
        {hover && hv != null && (
          <div
            className="chart__tip"
            style={{ left: Math.min(x(hv) + 12, width - 190), top: m.top + rowH * hover.g }}
          >
            <div className="chart__tip-row">
              <span className={`key-dot ${rows[hover.g].cls}`} aria-hidden="true" />
              <strong>{dec(hv)}</strong>
              <span>{hv >= data.threshold ? "ответ" : "отказ"}</span>
            </div>
            <div className="chart__tip-cm">{rows[hover.g].label}</div>
          </div>
        )}
      </div>
      {m.left <= 100 && <Legend items={rows.map((r) => ({ label: `${r.label} (${r.pts.length})`, cls: r.cls }))} />}
      <TableView caption="Сводка по группам">
        <thead>
          <tr>
            <th>группа</th>
            <th>запросов</th>
            <th>мин.</th>
            <th>макс.</th>
            <th>выше порога</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>{r.label}</td>
              <td>{r.pts.length}</td>
              <td>{dec(Math.min(...r.pts))}</td>
              <td>{dec(Math.max(...r.pts))}</td>
              <td>{r.above}</td>
            </tr>
          ))}
        </tbody>
      </TableView>
    </figure>
  );
}

/** Доли одного целого одной полосой: сегменты с зазором, подписи внутри, если влезают. */
export function StackedBar(props: {
  title: string;
  sub: string;
  total: number;
  parts: { label: string; value: number; tone: "ok" | "warn" | "bad" }[];
}) {
  return (
    <figure className="chart card">
      <figcaption className="chart__head">
        <span className="chart__title">{props.title}</span>
        <span className="chart__sub">{props.sub}</span>
      </figcaption>
      <div className="stack" role="img" aria-label={props.parts.map((p) => `${p.label}: ${p.value}`).join("; ")}>
        {props.parts.map((p) => (
          <div
            key={p.label}
            className={`stack__seg stack__seg--${p.tone}`}
            style={{ flexGrow: p.value }}
            title={`${p.label}: ${p.value} (${pct(p.value / props.total)})`}
          >
            <span>{p.value}</span>
          </div>
        ))}
      </div>
      <ul className="legend legend--stack">
        {props.parts.map((p) => (
          <li key={p.label}>
            <span className={`key-box stack__seg--${p.tone}`} aria-hidden="true" />
            <strong>{pct(p.value / props.total)}</strong> {p.label}
          </li>
        ))}
      </ul>
    </figure>
  );
}

/** Горизонтальные столбики одной серии: значение у конца столбика. */
export function Bars(props: { title: string; sub: string; unit: string; items: { label: string; value: number; note?: string }[] }) {
  const max = Math.max(...props.items.map((i) => i.value));
  return (
    <figure className="chart card">
      <figcaption className="chart__head">
        <span className="chart__title">{props.title}</span>
        <span className="chart__sub">{props.sub}</span>
      </figcaption>
      <ul className="hbars">
        {props.items.map((it) => (
          <li key={it.label} className="hbars__row">
            <span className="hbars__label">
              {it.label}
              {it.note && <span className="hbars__note">{it.note}</span>}
            </span>
            <span className="hbars__track">
              <span className="hbars__bar s1" style={{ width: `${(it.value / max) * 100}%` }} />
              <span className="hbars__value">
                {int(it.value)} {props.unit}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

/**
 * Показатель против шкалы баллов жюри: зоны «0 баллов / линейно / полный балл»
 * и отметка нашего значения. Цвет зон дублируется подписями.
 */
export function ScoreMeter(props: {
  label: string;
  value: number;
  unit: string;
  max: number;
  zones: [number, number];
  lowerIsBetter: boolean;
  note: string;
}) {
  const [a, b] = props.zones;
  const p = (v: number) => `${(Math.min(v, props.max) / props.max) * 100}%`;
  // Совпавшие границы — не шкала баллов, а жёсткий лимит (веса: превышение — недопуск).
  const zone = a === b
    ? [{ from: 0, to: a, cls: "full", text: `до ${a} — в пределах лимита` }]
    : props.lowerIsBetter
    ? [
        { from: 0, to: a, cls: "full", text: `≤ ${a} — полный балл` },
        { from: a, to: b, cls: "part", text: "линейно" },
        { from: b, to: props.max, cls: "zero", text: `> ${b} — 0` },
      ]
    : [
        { from: 0, to: a, cls: "zero", text: `< ${a} — 0` },
        { from: a, to: b, cls: "part", text: "линейно" },
        { from: b, to: props.max, cls: "full", text: `≥ ${b} — полный балл` },
      ];
  return (
    <div className="smeter">
      <div className="smeter__head">
        <span className="smeter__label">{props.label}</span>
        <span className="smeter__value">
          {String(props.value).replace(".", ",")}
          <small> {props.unit}</small>
        </span>
      </div>
      <div className="smeter__track" aria-hidden="true">
        {zone.map((z) => (
          <span key={z.cls} className={`smeter__zone smeter__zone--${z.cls}`} style={{ left: p(z.from), width: `calc(${p(z.to)} - ${p(z.from)})` }} />
        ))}
        <span className="smeter__pin" style={{ left: p(props.value) }} />
      </div>
      <div className="smeter__zones">
        {zone.map((z) => (
          <span key={z.cls} className={`smeter__zone-label smeter__zone-label--${z.cls}`}>
            {z.text}
          </span>
        ))}
      </div>
      <p className="smeter__note">{props.note}</p>
    </div>
  );
}

function Legend({ items }: { items: { label: string; cls: string; line?: boolean }[] }) {
  return (
    <ul className="legend">
      {items.map((it) => (
        <li key={it.label}>
          <span className={`${it.line ? "key-line" : "key-dot"} ${it.cls}`} aria-hidden="true" />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function TableView({ caption, children }: { caption: string; children: ReactNode }) {
  return (
    <details className="table-view">
      <summary>Таблица значений</summary>
      <div className="table-view__scroll">
        <table>
          <caption className="visually-hidden">{caption}</caption>
          {children}
        </table>
      </div>
    </details>
  );
}
