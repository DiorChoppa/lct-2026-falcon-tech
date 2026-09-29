import { Ban, Camera as CameraIcon, ChevronRight, CircleDot, MapPin, Moon, Sun } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { PlaceHypothesis } from "../../lib/route/places";
import { clock, dayLabel, msk } from "../../lib/route/time";
import type { Rejected } from "../../lib/route/track";
import type { Camera, Sighting } from "../../lib/route/types";
import { dec, pct, plural } from "../../lib/format";

const KIND_LABEL: Record<Sighting["kind"], string> = {
  real: "реальный кадр",
  sim: "смоделировано",
  lookalike: "похожая машина",
};

export function PlacesPanel(props: { items: PlaceHypothesis[]; onFocus: (placeId: string) => void }) {
  return (
    <section className="rpanel card" aria-labelledby="places-title">
      <h2 id="places-title" className="rpanel__title">Места стоянок</h2>
      <ul className="places">
        {props.items.slice(0, 6).map((h) => (
          <li key={h.place.id}>
            <button type="button" className={`place place--${h.role}`} onClick={() => props.onFocus(h.place.id)}>
              <span className="place__icon" aria-hidden="true">
                {h.role === "night" ? <Moon /> : h.role === "day" ? <Sun /> : <MapPin />}
              </span>
              <span className="place__body">
                <span className="place__fact">{h.fact}</span>
                <span className="place__addr">{h.place.address}{h.place.city !== "Москва" ? `, ${h.place.city}` : ""}</span>
                {h.guess && h.confidence != null && (
                  <span className="place__guess">
                    <span>{h.guess}</span>
                    <strong>{pct(h.confidence)}</strong>
                  </span>
                )}
                {h.confidence != null && (
                  <span className="meter" aria-hidden="true">
                    <span className="meter__fill" style={{ width: `${h.confidence * 100}%` }} />
                  </span>
                )}
                <span className="place__explain">{h.explain}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      <p className="rpanel__note">
        Гипотезы — подсказка для проверки оператором, а не установленный факт. В продукте — доступ по ролям и
        журнал аудита каждого просмотра маршрута.
      </p>
    </section>
  );
}

export function RejectedPanel(props: { rejected: Rejected[]; cameras: Map<string, Camera> }) {
  return (
    <section className="rpanel card" aria-labelledby="rej-title">
      <h2 id="rej-title" className="rpanel__title">
        <Ban aria-hidden="true" /> Отсеяно проверкой «успел ли доехать»
      </h2>
      {props.rejected.length === 0 ? (
        <p className="muted">В топ-100 не нашлось похожей машины, которую пришлось бы отсеять.</p>
      ) : (
        props.rejected.map((r) => {
          const a = props.cameras.get(r.conflictWith.cameraId)!;
          const b = props.cameras.get(r.sighting.cameraId)!;
          return (
            <div key={r.sighting.id} className="rejected">
              <figure>
                <img src={r.conflictWith.cropUrl} alt="Кадр нашей машины" />
                <figcaption>наша · {clock(r.conflictWith.ts)} · {a.address}</figcaption>
              </figure>
              <figure>
                <img src={r.sighting.cropUrl} alt="Кадр похожей машины" />
                <figcaption>
                  похожая{r.sighting.vehicleId ? ` (vehicle_id ${r.sighting.vehicleId})` : ""} ·{" "}
                  {r.sighting.confidence != null ? `ReID ${pct(r.sighting.confidence)}` : ""} · {clock(r.sighting.ts)} · {b.address}
                </figcaption>
              </figure>
              <p className="rejected__why">
                Через {Math.max(0, Math.round(r.gapMin))} мин в {Math.round(r.km)} км по дорогам — нужна скорость{" "}
                {Math.round(r.needKmh)} км/ч. По внешности похожа, физически — не она.
              </p>
              <p className="muted">Показ механизма: время и место похожей машины задал симулятор.</p>
            </div>
          );
        })
      )}
    </section>
  );
}

export function Timeline(props: {
  accepted: Sighting[];
  cameras: Map<string, Camera>;
  day: string | null;
  onDay: (day: string | null) => void;
  onSighting: (id: string) => void;
}) {
  // Раскрытие дня и «Трек дня» — две отдельные кнопки: кнопка внутри <summary>
  // была бы вложенным интерактивным элементом.
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const toggle = (k: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  const days = new Map<string, Sighting[]>();
  for (const s of props.accepted) {
    const k = msk(s.ts).dateKey;
    days.set(k, [...(days.get(k) ?? []), s]);
  }
  const keys = [...days.keys()].sort().reverse();
  return (
    <section className="rpanel card" aria-labelledby="tl-title">
      <h2 id="tl-title" className="rpanel__title">Лента появлений</h2>
      <ol className="tl">
        {keys.map((k) => {
          const items = days.get(k)!;
          const real = items.filter((s) => s.kind === "real").length;
          const isOpen = open.has(k) || props.day === k;
          return (
            <li key={k} className="tl__day">
              <div className="tl__head">
                <button type="button" className="tl__toggle" aria-expanded={isOpen} onClick={() => toggle(k)}>
                  <ChevronRight aria-hidden="true" className="tl__chevron" />
                  <span className="tl__date">{dayLabel(items[0].ts)}</span>
                  <span className="muted">{items.length} {plural(items.length, ["появление", "появления", "появлений"])}</span>
                  {real > 0 && <span className="badge badge--brand">реальных {real}</span>}
                </button>
                <button
                  type="button"
                  className={`btn btn--sm tl__show${props.day === k ? " is-active" : ""}`}
                  aria-pressed={props.day === k}
                  aria-label={`Трек дня: ${dayLabel(items[0].ts)}`}
                  onClick={() => props.onDay(props.day === k ? null : k)}
                >
                  Трек дня
                </button>
              </div>
              {isOpen && (
                <ol className="tl__items">
                  {items.map((s) => (
                    <li key={s.id}>
                      <button type="button" className="tl__item" onClick={() => props.onSighting(s.id)}>
                        <span className="mono">{clock(s.ts)}</span>
                        <span className={`tl__kind tl__kind--${s.kind}`} role="img" aria-label={KIND_LABEL[s.kind]}>
                          {s.kind === "real" ? <CameraIcon aria-hidden="true" /> : <CircleDot aria-hidden="true" />}
                        </span>
                        <span className="tl__addr">{props.cameras.get(s.cameraId)?.address}</span>
                        {s.dwellMin ? <span className="muted">{dec(s.dwellMin / 60, 1)} ч</span> : null}
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/** Карточка выбранного появления или камеры — поверх карты. */
export function SightingCard(props: { items: Sighting[]; camera: Camera; onClose: () => void }) {
  const last = props.items[props.items.length - 1];
  const ref = useRef<HTMLDivElement>(null);
  // Фокус — в карточку: так её видно и на узком экране, где карта выше ленты.
  useEffect(() => ref.current?.focus(), [props.items]);
  return (
    <div
      ref={ref}
      className="scard card"
      role="group"
      aria-label={props.camera.address}
      tabIndex={-1}
      onKeyDown={(e) => e.key === "Escape" && props.onClose()}
    >
      <button type="button" className="btn btn--ghost btn--sm btn--icon scard__close" onClick={props.onClose} aria-label="Закрыть">
        ×
      </button>
      <img src={last.cropUrl} alt="" />
      {last.kind === "sim" && <span className="scard__illus">иллюстративный кадр</span>}
      <p className="scard__addr">{props.camera.address}</p>
      <p className="muted">
        {props.camera.city} · {props.camera.kind === "parking" ? "парковочный комплекс" : "дорожная камера"}
      </p>
      <ul className="scard__list">
        {props.items.slice(-5).reverse().map((s) => (
          <li key={s.id}>
            <span className="mono">{dayLabel(s.ts)} {clock(s.ts)}</span> · {KIND_LABEL[s.kind]}
            {s.confidence != null && ` · ReID ${pct(s.confidence)}`}
            {s.dwellMin ? ` · стоянка ${dec(s.dwellMin / 60, 1)} ч` : ""}
          </li>
        ))}
      </ul>
      {props.items.length > 5 && <p className="muted">и ещё {props.items.length - 5}</p>}
    </div>
  );
}
