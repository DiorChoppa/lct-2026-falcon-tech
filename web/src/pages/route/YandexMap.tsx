import { useEffect, useRef, useState } from "react";
import { YMAPS_KEY } from "../../app/config";
import type { Theme } from "../../app/theme";
import type { LngLat, MapLayers, MapPlace, MapPoint } from "./layers";

// Минимальная типизация ymaps3 — ровно то, чем пользуемся ниже.
type Props = Record<string, unknown>;
interface YMapInstance {
  addChild(e: unknown): void;
  removeChild(e: unknown): void;
  update(p: Props): void;
  destroy(): void;
}
interface Ymaps3 {
  ready: Promise<void>;
  YMap: new (el: HTMLElement, props: Props) => YMapInstance;
  YMapDefaultSchemeLayer: new (p: Props) => unknown;
  YMapDefaultFeaturesLayer: new (p: Props) => unknown;
  YMapMarker: new (p: Props, el: HTMLElement) => unknown;
  YMapFeature: new (p: Props) => unknown;
}
declare global {
  interface Window {
    ymaps3?: Ymaps3;
  }
}

let loading: Promise<Ymaps3> | null = null;

/** Скрипт JS API 3 — один раз на страницу. */
function loadYmaps(key: string): Promise<Ymaps3> {
  loading ??= new Promise<Ymaps3>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = `https://api-maps.yandex.ru/v3/?apikey=${encodeURIComponent(key)}&lang=ru_RU`;
    s.async = true;
    const fail = () => {
      loading = null;
      reject(new Error("Яндекс Карты не загрузились"));
    };
    // Скрипт с неверным ключом загружается, но ymaps3 не объявляет — иначе вечная загрузка
    s.onload = () => {
      const api = window.ymaps3;
      if (api) api.ready.then(() => resolve(api), fail);
      else fail();
    };
    s.onerror = fail;
    document.head.appendChild(s);
  });
  return loading;
}

/** ymaps3 ждёт границы как [левый верхний, правый нижний] в [lon, lat]. */
const toYBounds = ([[minLon, minLat], [maxLon, maxLat]]: [LngLat, LngLat]): [LngLat, LngLat] => [
  [minLon, maxLat],
  [maxLon, minLat],
];

type Status = "nokey" | "loading" | "ready" | "error";

export function YandexMap(props: {
  layers: MapLayers;
  theme: Theme;
  /** id меток, которые подсветить: выбранное появление и его камера. */
  highlight: string[];
  onPoint: (id: string) => void;
  onPlace: (id: string) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const inst = useRef<{ api: Ymaps3; map: YMapInstance; children: unknown[] } | null>(null);
  const handlers = useRef(props);
  handlers.current = props;
  const [status, setStatus] = useState<Status>(YMAPS_KEY ? "loading" : "nokey");

  useEffect(() => {
    if (!YMAPS_KEY || !box.current) return;
    let alive = true;
    loadYmaps(YMAPS_KEY)
      .then((api) => {
        if (!alive || !box.current) return;
        const map = new api.YMap(box.current, {
          location: { bounds: toYBounds(handlers.current.layers.bounds) },
          theme: handlers.current.theme,
          margin: [48, 48, 48, 48],
        });
        map.addChild(new api.YMapDefaultSchemeLayer({}));
        map.addChild(new api.YMapDefaultFeaturesLayer({}));
        inst.current = { api, map, children: [] };
        setStatus("ready");
      })
      .catch(() => alive && setStatus("error"));
    return () => {
      alive = false;
      inst.current?.map.destroy();
      inst.current = null;
    };
  }, []);

  useEffect(() => {
    inst.current?.map.update({ theme: props.theme });
  }, [props.theme, status]);

  useEffect(() => {
    const m = inst.current;
    if (!m) return;
    m.children.forEach((c) => m.map.removeChild(c));
    m.children = [];
    const add = (c: unknown) => {
      m.map.addChild(c);
      m.children.push(c);
    };
    const { layers } = props;
    if (layers.line) {
      add(new m.api.YMapFeature({
        geometry: { type: "LineString", coordinates: layers.line },
        // Жёлтый теряется на жёлтых дорогах светлой схемы — там оранжевый.
        style: { stroke: [{ color: props.theme === "dark" ? "#ffe14d" : "#eb6834", width: 4 }] },
      }));
    }
    for (const p of [...layers.points, ...layers.rejected]) {
      const on = props.highlight.includes(p.id);
      const el = pointEl(p);
      el.classList.toggle("is-selected", on);
      add(new m.api.YMapMarker({ coordinates: [p.lon, p.lat], zIndex: on ? 20 : undefined, onClick: () => handlers.current.onPoint(p.id) }, el));
    }
    for (const p of layers.places) {
      add(new m.api.YMapMarker({ coordinates: [p.lon, p.lat], zIndex: 10, onClick: () => handlers.current.onPlace(p.id) }, placeEl(p)));
    }
  }, [props.layers, props.highlight.join(), props.theme, status]);

  useEffect(() => {
    inst.current?.map.update({ location: { bounds: toYBounds(props.layers.bounds), duration: 500 } });
  }, [props.layers.bounds, status]);

  if (status === "nokey" || status === "error") {
    return (
      <div className="ymap ymap--stub" role="note">
        <p className="ymap__stub-title">{status === "nokey" ? "Ключ Яндекс Карт не задан" : "Яндекс Карты не загрузились"}</p>
        <p className="muted">
          {status === "nokey"
            ? "Задайте YMAPS_KEY в .env стенда (или VITE_YMAPS_KEY в web/.env.local для разработки). Места, лента и сводки работают и без карты."
            : "Проверьте ключ и ограничение по доменам в кабинете разработчика. Места, лента и сводки работают и без карты."}
        </p>
      </div>
    );
  }
  return <div ref={box} className="ymap" aria-label="Карта появлений" role="region" aria-busy={status === "loading"} />;
}

// Метки — обычные DOM-узлы: ymaps3 кладёт их в свой слой. Текст — только через textContent.
function pointEl(p: MapPoint): HTMLElement {
  const el = document.createElement("button");
  el.type = "button";
  el.className = `rm rm--${p.kind}`;
  el.title = p.kind === "lookalike" ? `Отсеяно: ${p.label}` : `${p.label} · ${p.count}`;
  el.setAttribute("aria-label", el.title);
  const text = p.order ?? (p.count > 1 ? p.count : "");
  el.textContent = String(text);
  return el;
}

function placeEl(p: MapPlace): HTMLElement {
  const el = document.createElement("button");
  el.type = "button";
  el.className = `rm-place rm-place--${p.role}`;
  el.title = `${p.title} · ${Math.round(p.confidence * 100)} %`;
  el.setAttribute("aria-label", el.title);
  const icon = document.createElement("span");
  icon.className = "rm-place__icon";
  icon.textContent = p.role === "night" ? "☾" : "☀";
  const pct = document.createElement("span");
  pct.textContent = `${Math.round(p.confidence * 100)}%`;
  el.append(icon, pct);
  return el;
}
