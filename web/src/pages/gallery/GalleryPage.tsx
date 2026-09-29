import { AlertTriangle, ChevronLeft, ChevronRight, FileUp, ImageOff, Pencil, Plus, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { listGallery, setPlate } from "../../api/client";
import type { GalleryItem, GalleryPage as Page } from "../../api/types";
import { usePageFocus } from "../../app/usePageFocus";
import { int, plural } from "../../lib/format";
import { AddDialog, ImportDialog } from "./GalleryDialogs";
import "./gallery.css";

const PAGE_SIZES = [24, 48, 96];

type State = { kind: "loading"; prev: Page | null } | { kind: "done"; page: Page } | { kind: "error"; message: string };

/** Номер страницы и размер — в адресе: ссылка на страницу галереи открывается там же. */
function readQuery() {
  const q = new URLSearchParams(window.location.search);
  const page = Math.max(1, Number(q.get("page")) || 1);
  const size = PAGE_SIZES.includes(Number(q.get("size"))) ? Number(q.get("size")) : PAGE_SIZES[0];
  return { page, size };
}

function writeQuery(page: number, size: number) {
  const q = new URLSearchParams(window.location.search);
  q.set("page", String(page));
  q.set("size", String(size));
  window.history.replaceState(null, "", `${window.location.pathname}?${q}`);
}

export default function GalleryPage() {
  const titleRef = usePageFocus<HTMLHeadingElement>("Галерея");
  const [{ page, size }, setQuery] = useState(readQuery);
  const [state, setState] = useState<State>({ kind: "loading", prev: null });
  const [dialog, setDialog] = useState<"add" | "import" | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    writeQuery(page, size);
    const ctrl = new AbortController();
    setState((s) => ({ kind: "loading", prev: s.kind === "done" ? s.page : s.kind === "loading" ? s.prev : null }));
    listGallery(page, size, ctrl.signal)
      .then((p) => setState({ kind: "done", page: p }))
      .catch((e) => {
        if ((e as Error).name !== "AbortError") setState({ kind: "error", message: (e as Error).message });
      });
    return () => ctrl.abort();
  }, [page, size, reload]);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(t);
  }, [toast]);

  const shown = state.kind === "done" ? state.page : state.kind === "loading" ? state.prev : null;
  const totalPages = shown?.pagination.totalPages ?? 1;
  const go = (p: number) => {
    setQuery({ page: Math.min(Math.max(1, p), Math.max(1, totalPages)), size });
    titleRef.current?.scrollIntoView({ block: "start" });
  };

  const onItem = useCallback(
    (next: GalleryItem) =>
      setState((s) =>
        s.kind === "done"
          ? { kind: "done", page: { ...s.page, data: s.page.data.map((it) => (it.id === next.id ? next : it)) } }
          : s,
      ),
    [],
  );

  return (
    <div className="page gallery">
      <div className="page__head">
        <div>
          <p className="eyebrow">База снимков · пополнение и ГРЗ</p>
          <h1 ref={titleRef} tabIndex={-1}>
            Галерея ТС
          </h1>
        </div>
        <p className="page__lead">
          Записи, среди которых идёт поиск: кроп по bbox, эмбеддинг и метаданные. Известный номер из другого
          источника (ALPR другой камеры, ручной ввод) привязывается к записи и показывается при совпадении.
        </p>
      </div>

      <div className="toolbar card">
        <div className="toolbar__stat">
          <span className="toolbar__count">{shown ? int(shown.pagination.totalItems) : "—"}</span>
          <span className="muted">
            {shown ? plural(shown.pagination.totalItems, ["запись", "записи", "записей"]) : "записей"} в галерее
          </span>
        </div>
        <div className="toolbar__actions">
          <button type="button" className="btn" onClick={() => setDialog("import")}>
            <FileUp aria-hidden="true" />
            Импорт CSV
          </button>
          <button type="button" className="btn btn--primary" onClick={() => setDialog("add")}>
            <Plus aria-hidden="true" />
            Добавить ТС
          </button>
        </div>
      </div>

      {state.kind === "error" && (
        <div className="alert alert--error" role="alert">
          <AlertTriangle aria-hidden="true" />
          <div>
            <strong>Галерея не загрузилась.</strong> {state.message}
            <div className="alert__actions">
              <button type="button" className="btn btn--sm" onClick={() => setReload((n) => n + 1)}>
                Повторить
              </button>
            </div>
          </div>
        </div>
      )}

      {!shown && state.kind === "loading" && (
        <ul className="g-grid" aria-hidden="true">
          {Array.from({ length: 12 }, (_, i) => (
            <li key={i} className="g-card card">
              <div className="skeleton" style={{ aspectRatio: "4 / 3", borderRadius: 0 }} />
              <div className="g-card__body">
                <div className="skeleton" style={{ height: 16, width: "60%" }} />
                <div className="skeleton" style={{ height: 36 }} />
              </div>
            </li>
          ))}
        </ul>
      )}

      {shown && shown.data.length === 0 && (
        <div className="empty card">
          <ImageOff aria-hidden="true" />
          <h2>Галерея пуста</h2>
          <p className="muted">Добавьте ТС вручную или импортируйте разметку датасета одним CSV.</p>
        </div>
      )}

      {shown && shown.data.length > 0 && (
        <>
          <ul className={`g-grid${state.kind === "loading" ? " is-refetching" : ""}`} aria-busy={state.kind === "loading"}>
            {shown.data.map((it) => (
              <GalleryCard key={it.id} item={it} onChange={onItem} />
            ))}
          </ul>
          <Pager
            page={page}
            totalPages={totalPages}
            size={size}
            onPage={go}
            onSize={(s) => setQuery({ page: 1, size: s })}
          />
        </>
      )}

      {dialog === "add" && (
        <AddDialog
          onClose={() => setDialog(null)}
          onDone={(item) => {
            setToast(`Запись #${item.id} добавлена в галерею`);
            setQuery({ page: 1, size });
            setReload((n) => n + 1);
          }}
        />
      )}
      {dialog === "import" && (
        <ImportDialog
          onClose={() => setDialog(null)}
          onDone={(r) => {
            setToast(`Импортировано ${int(r.imported)}, с ошибкой ${int(r.failed)}`);
            setQuery({ page: 1, size });
            setReload((n) => n + 1);
          }}
        />
      )}

      <div className="toast-region" aria-live="polite">
        {toast && <div className="toast">{toast}</div>}
      </div>
    </div>
  );
}

function Pager(props: {
  page: number;
  totalPages: number;
  size: number;
  onPage: (p: number) => void;
  onSize: (s: number) => void;
}) {
  const { page, totalPages } = props;
  return (
    <nav className="pager" aria-label="Страницы галереи">
      <button type="button" className="btn btn--sm" onClick={() => props.onPage(page - 1)} disabled={page <= 1}>
        <ChevronLeft aria-hidden="true" />
        Назад
      </button>
      <span className="pager__pos num">
        стр. <strong>{int(page)}</strong> из {int(totalPages)}
      </span>
      <button
        type="button"
        className="btn btn--sm"
        onClick={() => props.onPage(page + 1)}
        disabled={page >= totalPages}
      >
        Вперёд
        <ChevronRight aria-hidden="true" />
      </button>
      <label className="pager__size">
        <span className="muted">на странице</span>
        <select className="input" value={props.size} onChange={(e) => props.onSize(Number(e.target.value))}>
          {PAGE_SIZES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
    </nav>
  );
}

const PLATE_MAX = 16;

function GalleryCard({ item, onChange }: { item: GalleryItem; onChange: (i: GalleryItem) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(item.plate ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async (plate: string | null) => {
    setBusy(true);
    setError(null);
    try {
      onChange(await setPlate(item.id, plate));
      setEditing(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="g-card card">
      <div className="g-card__media">
        <img src={item.cropUrl} alt={`Запись ${item.id}${item.vehicleId ? `, vehicle_id ${item.vehicleId}` : ""}`} loading="lazy" />
        <span className="g-card__id mono">#{item.id}</span>
      </div>
      <div className="g-card__body">
        <dl className="g-card__meta">
          <div>
            <dt>vehicle_id</dt>
            <dd className="mono">{item.vehicleId ?? <span className="muted">—</span>}</dd>
          </div>
          <div>
            <dt>кадр</dt>
            <dd className="mono g-card__image" title={item.imageId}>
              {item.imageId}
            </dd>
          </div>
        </dl>
        {editing ? (
          <form
            className="plate-form"
            onSubmit={(e) => {
              e.preventDefault();
              const v = value.trim().toUpperCase();
              save(v === "" ? null : v);
            }}
          >
            <label className="visually-hidden" htmlFor={`plate-${item.id}`}>
              ГРЗ для записи {item.id}
            </label>
            <input
              id={`plate-${item.id}`}
              className="input input--mono plate-form__input"
              value={value}
              maxLength={PLATE_MAX}
              autoComplete="off"
              spellCheck={false}
              placeholder="А123ВС777"
              autoFocus
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && setEditing(false)}
            />
            <button type="submit" className="btn btn--primary btn--sm" disabled={busy} aria-busy={busy}>
              {busy ? <span className="spinner" aria-hidden="true" /> : "OK"}
            </button>
            <button
              type="button"
              className="btn btn--ghost btn--sm btn--icon"
              onClick={() => setEditing(false)}
              aria-label="Отмена"
            >
              <X aria-hidden="true" />
            </button>
            {error && (
              <p className="field__error plate-form__error" role="alert">
                {error}
              </p>
            )}
            <p className="field__hint plate-form__hint">Пусто — снять привязку</p>
          </form>
        ) : (
          <div className="g-card__plate">
            {item.plate ? <span className="plate">{item.plate}</span> : <span className="muted">без ГРЗ</span>}
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => {
                setValue(item.plate ?? "");
                setEditing(true);
              }}
              aria-label={item.plate ? `Изменить ГРЗ записи ${item.id}` : `Привязать ГРЗ к записи ${item.id}`}
            >
              <Pencil aria-hidden="true" />
              {item.plate ? "Изменить" : "Привязать"}
            </button>
          </div>
        )}
      </div>
    </li>
  );
}
