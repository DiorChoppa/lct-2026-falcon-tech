import { ImageUp, RotateCcw } from "lucide-react";
import { useEffect, useId, useRef, useState, type DragEvent, type PointerEvent } from "react";
import type { Bbox } from "../api/types";
import { clampBbox, fromCorners, isUsable, toDisplay, toNatural, type Scale } from "../lib/bbox";

interface Props {
  fileUrl: string | null;
  fileName?: string;
  fileError: string | null;
  bbox: Bbox | null;
  onFile: (f: File | null) => void;
  onBbox: (b: Bbox | null) => void;
}

/**
 * Кадр + выделение ТС: мышью/пальцем по кадру или числами (альтернатива
 * перетаскиванию для клавиатуры). bbox всегда в пикселях исходного кадра.
 */
export function FramePicker({ fileUrl, fileName, fileError, bbox, onFile, onBbox }: Props) {
  const [scale, setScale] = useState<Scale | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const [dropping, setDropping] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const errorId = useId();

  // Масштаб отображения ↔ исходные пиксели; ResizeObserver ловит и resize окна, и смену раскладки.
  useEffect(() => {
    const img = imgRef.current;
    if (!img) {
      setScale(null);
      return;
    }
    const measure = () => {
      if (!img.naturalWidth) return;
      setScale({
        displayW: img.clientWidth,
        displayH: img.clientHeight,
        naturalW: img.naturalWidth,
        naturalH: img.naturalHeight,
      });
    };
    const ro = new ResizeObserver(measure);
    ro.observe(img);
    img.addEventListener("load", measure);
    measure();
    return () => {
      ro.disconnect();
      img.removeEventListener("load", measure);
    };
  }, [fileUrl]);

  const pointToNatural = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return toNatural(e.clientX - rect.left, e.clientY - rect.top, scale!);
  };
  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!scale || e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = pointToNatural(e);
    setDrag(p);
    onBbox({ x: p.x, y: p.y, w: 0, h: 0 });
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!drag || !scale) return;
    onBbox(fromCorners(drag, pointToNatural(e)));
  };
  const onUp = (e: PointerEvent<HTMLDivElement>) => {
    if (!drag || !scale) return;
    onBbox(clampBbox(fromCorners(drag, pointToNatural(e)), scale.naturalW, scale.naturalH));
    setDrag(null);
  };

  const setField = (k: keyof Bbox, v: string) => {
    if (!scale) return;
    const n = Number(v);
    if (!Number.isFinite(n)) return;
    const next = { ...(bbox ?? { x: 0, y: 0, w: 0, h: 0 }), [k]: Math.round(n) };
    onBbox(clampBbox(next, scale.naturalW, scale.naturalH));
  };

  const dropHandlers = {
    onDragOver: (e: DragEvent) => {
      if (!e.dataTransfer.types.includes("Files")) return;
      e.preventDefault();
      setDropping(true);
    },
    onDragLeave: () => setDropping(false),
    onDrop: (e: DragEvent) => {
      e.preventDefault();
      setDropping(false);
      const f = e.dataTransfer.files[0];
      if (f) onFile(f);
    },
  };

  const shown = bbox && scale ? toDisplay(bbox, scale) : null;
  const tooSmall = !!bbox && bbox.w > 0 && !isUsable(bbox) && !drag;

  const fileInput = (
    <input
      ref={inputRef}
      id={inputId}
      tabIndex={fileUrl ? -1 : undefined}
      className="visually-hidden"
      type="file"
      accept="image/jpeg,image/png"
      aria-describedby={fileError ? errorId : undefined}
      onChange={(e) => {
        onFile(e.target.files?.[0] ?? null);
        e.target.value = "";
      }}
    />
  );

  return (
    <div className="picker">
      {fileUrl ? (
        <>
          <div
            className={`picker__stage${dropping ? " is-dropping" : ""}`}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={() => setDrag(null)}
            {...dropHandlers}
            role="img"
            aria-label="Кадр с камеры. Протяните по кадру, чтобы выделить транспортное средство, или введите координаты ниже."
          >
            <img ref={imgRef} src={fileUrl} alt="" draggable={false} />
            {shown && shown.w > 0 && (
              <div
                className={`picker__box reticle${drag ? " is-drawing" : ""}`}
                style={{ left: shown.x, top: shown.y, width: shown.w, height: shown.h }}
              >
                {bbox && !drag && (
                  <span className="picker__size mono">
                    {bbox.w}×{bbox.h}
                  </span>
                )}
              </div>
            )}
            {!bbox && scale && (
              <div className="picker__hint" aria-hidden="true">
                Протяните рамку вокруг автомобиля
              </div>
            )}
          </div>
          <div className="picker__bar">
            <span className="picker__file mono" title={fileName}>
              {fileName}
              {scale && ` · ${scale.naturalW}×${scale.naturalH}`}
            </span>
            <span className="picker__actions">
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => onBbox(null)}
                disabled={!bbox}
              >
                <RotateCcw aria-hidden="true" />
                Сбросить рамку
              </button>
              <button type="button" className="btn btn--sm" onClick={() => inputRef.current?.click()}>
                <ImageUp aria-hidden="true" />
                Другой кадр
              </button>
              {fileInput}
            </span>
          </div>
        </>
      ) : (
        <label htmlFor={inputId} className={`dropzone${dropping ? " is-dropping" : ""}`} {...dropHandlers}>
          <span className="dropzone__icon reticle" aria-hidden="true">
            <ImageUp />
          </span>
          <span className="dropzone__title">Перетащите кадр с камеры</span>
          <span className="dropzone__sub">
            или <span className="dropzone__link">выберите файл</span>, или вставьте из буфера <kbd>Ctrl</kbd>+
            <kbd>V</kbd>
          </span>
          <span className="dropzone__meta">JPEG или PNG до 3 МиБ · полный кадр, не кроп</span>
          {fileInput}
        </label>
      )}

      {fileError && (
        <p id={errorId} className="field__error" role="alert">
          {fileError}
        </p>
      )}

      <div className="picker__coords">
        <fieldset className="bbox-fields" disabled={!scale}>
          <legend className="field__label">BBox, пиксели кадра</legend>
          {(["x", "y", "w", "h"] as const).map((k) => (
            <label key={k} className="bbox-fields__item">
              <span className="mono">{k}</span>
              <input
                className="input input--mono"
                type="number"
                min={0}
                inputMode="numeric"
                value={bbox ? bbox[k] : ""}
                onChange={(e) => setField(k, e.target.value)}
              />
            </label>
          ))}
        </fieldset>
        <CropPreview url={fileUrl} bbox={isUsable(bbox) ? bbox : null} scale={scale} />
      </div>
      {tooSmall && (
        <p className="field__error" role="alert">
          Рамка меньше 8×8 пикселей — выделите автомобиль целиком.
        </p>
      )}
    </div>
  );
}

/** Что уйдёт в модель: кроп по bbox, нарисованный фоном без canvas. */
function CropPreview({ url, bbox, scale }: { url: string | null; bbox: Bbox | null; scale: Scale | null }) {
  if (!url || !bbox || !scale) {
    return (
      <div className="crop-preview crop-preview--empty">
        <span>кроп</span>
      </div>
    );
  }
  const { naturalW: W, naturalH: H } = scale;
  const pos = (off: number, size: number, total: number) => (total === size ? 0 : (off / (total - size)) * 100);
  return (
    <div
      className="crop-preview reticle"
      role="img"
      aria-label="Кроп по рамке — это видит модель"
      style={{
        aspectRatio: `${bbox.w} / ${bbox.h}`,
        backgroundImage: `url(${url})`,
        backgroundSize: `${(W / bbox.w) * 100}% ${(H / bbox.h) * 100}%`,
        backgroundPosition: `${pos(bbox.x, bbox.w, W)}% ${pos(bbox.y, bbox.h, H)}%`,
      }}
    />
  );
}
