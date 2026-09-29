import type { CSSProperties } from "react";
import type { CityStat } from "../../lib/route/summary";
import { dayLabel } from "../../lib/route/time";

const DAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"];

/** Когда машину видят: 7 × 24, одна шкала одного цвета; табличный вид — в details. */
export function HeatStrip({ grid }: { grid: number[][] }) {
  const max = Math.max(1, ...grid.flat());
  return (
    <figure className="heat card">
      <figcaption className="chart__head">
        <span className="chart__title">Когда машину видят</span>
        <span className="chart__sub">Появления по дням недели и часам, время московское</span>
      </figcaption>
      <div className="heat__grid" role="img" aria-label="Тепловая карта появлений по дням недели и часам">
        <span />
        {Array.from({ length: 24 }, (_, h) => (
          <span key={h} className="heat__hour">{h % 3 === 0 ? h : ""}</span>
        ))}
        {grid.map((row, d) => (
          <div key={d} className="heat__row">
            <span className="heat__day">{DAYS[d]}</span>
            {row.map((v, h) => (
              <span
                key={h}
                className="heat__cell"
                style={{ "--v": v / max } as CSSProperties}
                title={`${DAYS[d]}, ${h}:00 — ${v}`}
              />
            ))}
          </div>
        ))}
      </div>
      <details className="table-view">
        <summary>Таблица значений</summary>
        <div className="table-view__scroll">
          <table>
            <thead>
              <tr>
                <th>день</th>
                {Array.from({ length: 24 }, (_, h) => <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {grid.map((row, d) => (
                <tr key={d}>
                  <td>{DAYS[d]}</td>
                  {row.map((v, h) => <td key={h}>{v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

export function Cities({ items }: { items: CityStat[] }) {
  return (
    <section className="cities card" aria-labelledby="cities-title">
      <h2 id="cities-title" className="rpanel__title">Города</h2>
      <ul>
        {items.map((c) => (
          <li key={c.city}>
            <span className="cities__name">{c.city}</span>
            <span className="cities__count">{c.count}</span>
            <span className="muted">
              {dayLabel(c.from)}
              {c.to - c.from > 86_400_000 ? ` — ${dayLabel(c.to)}` : ""}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
