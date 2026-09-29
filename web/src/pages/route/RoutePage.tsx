import { AlertTriangle, ArrowLeft, FlaskConical } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Link, ROUTES } from "../../app/router";
import type { Theme } from "../../app/theme";
import { usePageFocus } from "../../app/usePageFocus";
import { int } from "../../lib/format";
import { dayLabel } from "../../lib/route/time";
import type { SearchSession } from "../search/useSearchSession";
import { toLayers, type Scope } from "./layers";
import { Cities, HeatStrip } from "./Insights";
import { PlacesPanel, RejectedPanel, SightingCard, Timeline } from "./Panels";
import { useRoute } from "./useRoute";
import { YandexMap } from "./YandexMap";
import "./route.css";
import "../solution/charts.css";

type Selection = { kind: "camera"; id: string } | { kind: "sighting"; id: string } | null;

export default function RoutePage(props: { vehicleId: string | null; session: SearchSession; theme: Theme }) {
  const titleRef = usePageFocus<HTMLHeadingElement>(`Маршрут ТС ${props.vehicleId ?? ""}`);
  const { state, retry } = useRoute(props.vehicleId, props.session);
  const [scope, setScope] = useState<Scope>("moscow");
  const [day, setDay] = useState<string | null>(null);
  const [selected, setSelected] = useState<Selection>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const mapRef = useRef<HTMLElement>(null);
  const opener = useRef<HTMLElement | null>(null);

  // На узком экране карта выше панелей: показать её, иначе клик по месту или
  // «Трек дня» ничего видимого не делает. После отрисовки: плашка дня меняет высоту карты.
  const showMap = () => {
    if (!window.matchMedia("(max-width: 1100px)").matches) return;
    requestAnimationFrame(() => mapRef.current?.scrollIntoView({ block: "nearest" }));
  };
  const select = (sel: Selection) => {
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setSelected(sel);
  };
  const closeCard = () => {
    setSelected(null);
    if (opener.current?.isConnected) opener.current.focus();
  };

  const data = state.kind === "done" ? state.data : null;
  const layers = useMemo(() => {
    if (!data) return null;
    const l = toLayers(data, { scope, day });
    const place = focus ? data.places.find((h) => h.place.id === focus) : null;
    if (place) {
      const d = 0.01;
      l.bounds = [[place.place.lon - d, place.place.lat - d], [place.place.lon + d, place.place.lat + d]];
    }
    return l;
  }, [data, scope, day, focus]);

  const card = useMemo(() => {
    if (!data || !selected) return null;
    const all = [...data.track.accepted, ...data.track.rejected.map((r) => r.sighting)];
    const byCam = selected.kind === "camera" && !day ? data.track.accepted.filter((s) => s.cameraId === selected.id) : [];
    // Красная метка двойника передаёт id появления, а не камеры
    const items = byCam.length ? byCam : all.filter((s) => s.id === selected.id);
    const cam = items[0] && data.cameras.get(items[0].cameraId);
    return items.length && cam ? { items, cam } : null;
  }, [data, selected, day]);

  return (
    <div className="page route">
      <div className="route__head">
        <Link href={ROUTES.search} className="route__back">
          <ArrowLeft aria-hidden="true" /> К поиску
        </Link>
        <p className="eyebrow">Развитие продукта · маршрут и места стоянок</p>
        <h1 ref={titleRef} tabIndex={-1}>
          Маршрут ТС · vehicle_id {props.vehicleId}
        </h1>
        {data && (
          <p className="muted">
            {dayLabel(data.from)} — {dayLabel(data.to - 1)} · {state.kind === "done" && state.source === "scenario" ? "запрос из сценария демо" : "запрос из текущего поиска"}
          </p>
        )}
      </div>

      <p className="sim-banner" role="note">
        <FlaskConical aria-hidden="true" />
        <span>
          <strong>Симуляция.</strong> Камеры, время и история смоделированы: по ТЗ (§5.2) их нет в данных, а анонимный
          camera_id датасета не используем. Реальны кадры, найденные моделью, и их уверенность. На ответ поиска и метрику
          маршрут не влияет.
        </span>
      </p>

      {state.kind === "loading" && (
        <div className="route__grid" aria-busy="true">
          <div className="skeleton" style={{ minHeight: 480 }} />
          <div className="skeleton" style={{ minHeight: 480 }} />
        </div>
      )}

      {state.kind === "error" && (
        <div className="alert alert--error" role="alert">
          <AlertTriangle aria-hidden="true" />
          <div>
            <strong>Маршрут не собран.</strong> {state.message}
            <div className="alert__actions">
              {state.retry && (
                <button type="button" className="btn btn--primary btn--sm" onClick={retry}>
                  Повторить
                </button>
              )}
              <Link href={ROUTES.search} className="btn btn--sm">
                К поиску
              </Link>
            </div>
          </div>
        </div>
      )}

      {data && layers && (
        <>
          <dl className="route__stats">
            <Stat label="появлений" value={int(data.stats.sightings)} />
            <Stat label="реальных кадров" value={int(data.stats.real)} />
            <Stat label="регулярных мест" value={int(data.stats.regular)} />
            <Stat label="городов" value={int(data.stats.cities)} />
            <Stat label="отсеяно" value={int(data.stats.rejected)} />
          </dl>

          <div className="route__grid">
            <section ref={mapRef} className="route__map card" aria-label="Карта">
              <div className="route__toolbar">
                <div className="seg" role="group" aria-label="Охват карты">
                  {(["moscow", "all"] as const).map((s) => (
                    <button key={s} type="button" aria-pressed={scope === s} onClick={() => { setScope(s); setFocus(null); }}>
                      {s === "moscow" ? "Москва" : "Вся поездка"}
                    </button>
                  ))}
                </div>
                {day && (
                  <button
                    type="button"
                    className="badge badge--brand route__day"
                    aria-label={`Скрыть трек за ${dayLabel(Date.parse(`${day}T12:00:00+03:00`))}`}
                    onClick={() => setDay(null)}
                  >
                    Трек за {dayLabel(Date.parse(`${day}T12:00:00+03:00`))} <span aria-hidden="true">×</span>
                  </button>
                )}
              </div>
              <YandexMap
                layers={layers}
                theme={props.theme}
                highlight={selected && card ? [selected.id, card.items[0].cameraId] : []}
                onPoint={(id) => select(day ? { kind: "sighting", id } : { kind: "camera", id })}
                onPlace={(id) => setFocus(id)}
              />
              {card && <SightingCard items={card.items} camera={card.cam} onClose={closeCard} />}
              <ul className="legend route__legend">
                <li><span className="rm rm--real rm--legend" aria-hidden="true" /> реальный кадр</li>
                <li><span className="rm rm--sim rm--legend" aria-hidden="true" /> смоделированное появление</li>
                <li><span className="rm rm--lookalike rm--legend" aria-hidden="true" /> отсеянная похожая машина</li>
              </ul>
            </section>

            <aside className="route__side">
              <PlacesPanel items={data.places} onFocus={(id) => { setFocus(id); showMap(); }} />
              <RejectedPanel rejected={data.track.rejected} cameras={data.cameras} />
              <Timeline
                accepted={data.track.accepted}
                cameras={data.cameras}
                day={day}
                onDay={(d) => { setDay(d); setFocus(null); setSelected(null); if (d) showMap(); }}
                onSighting={(id) => select({ kind: "sighting", id })}
              />
            </aside>
          </div>

          <div className="route__insights">
            <HeatStrip grid={data.heat} />
            <Cities items={data.cities} />
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="route__stat">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
