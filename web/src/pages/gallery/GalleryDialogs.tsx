import { AlertTriangle, CheckCircle2, X } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { addToGallery, checkImage, importGallery } from "../../api/client";
import type { Bbox, GalleryItem, ImportReport } from "../../api/types";
import { FramePicker } from "../../components/FramePicker";
import { isUsable } from "../../lib/bbox";
import { int } from "../../lib/format";

/** Модальное окно на нативном <dialog>: фокус внутри, Esc и клик по фону закрывают. */
function Modal(props: {
  title: string;
  eyebrow: string;
  narrow?: boolean;
  busy?: boolean;
  onClose: () => void;
  children: ReactNode;
  foot: ReactNode;
  onSubmit: (e: FormEvent) => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  // Во время отправки окно не закрываем — иначе результат потеряется молча.
  const close = () => !props.busy && ref.current?.close();
  return (
    <dialog
      ref={ref}
      className={`dialog${props.narrow ? " dialog--narrow" : ""}`}
      aria-labelledby="modal-title"
      onClose={props.onClose}
      onCancel={(e) => props.busy && e.preventDefault()}
      onClick={(e) => e.target === ref.current && close()}
    >
      <form onSubmit={props.onSubmit} className="dialog__form">
        <div className="dialog__head">
          <div>
            <p className="eyebrow">{props.eyebrow}</p>
            <h2 id="modal-title">{props.title}</h2>
          </div>
          <button type="button" className="btn btn--ghost btn--icon" onClick={close} aria-label="Закрыть">
            <X aria-hidden="true" />
          </button>
        </div>
        <div className="dialog__body">{props.children}</div>
        <div className="dialog__foot">{props.foot}</div>
      </form>
    </dialog>
  );
}

export function AddDialog({ onClose, onDone }: { onClose: () => void; onDone: (i: GalleryItem) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [vehicleId, setVehicleId] = useState("");
  const [plate, setPlateValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<GalleryItem | null>(null);

  useEffect(() => {
    if (!file) return;
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file || !isUsable(bbox)) return;
    setBusy(true);
    setError(null);
    try {
      const item = await addToGallery({
        image: file,
        bbox,
        vehicleId: vehicleId.trim() || undefined,
        plate: plate.trim().toUpperCase() || undefined,
      });
      setDone(item);
      onDone(item);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      eyebrow="Пополнение галереи"
      title="Добавить ТС"
      busy={busy}
      onClose={onClose}
      onSubmit={submit}
      foot={
        done ? (
          <button type="button" className="btn btn--primary" onClick={(e) => e.currentTarget.closest("dialog")?.close()}>
            Готово
          </button>
        ) : (
          <button type="submit" className="btn btn--primary" disabled={!file || !isUsable(bbox) || busy} aria-busy={busy}>
            {busy && <span className="spinner" aria-hidden="true" />}
            {busy ? "Считаем эмбеддинг…" : "Добавить в галерею"}
          </button>
        )
      }
    >
      {done ? (
        <div className="done-state">
          <CheckCircle2 aria-hidden="true" />
          <div>
            <strong>Запись #{done.id} добавлена.</strong>
            <p className="muted">Кроп сохранён, эмбеддинг посчитан — запись уже участвует в поиске.</p>
          </div>
          <img src={done.cropUrl} alt="Кроп новой записи" />
        </div>
      ) : (
        <>
          <FramePicker
            fileUrl={url}
            fileName={file?.name}
            fileError={fileError}
            bbox={bbox}
            onFile={(f) => {
              const err = f ? checkImage(f) : null;
              setFileError(err);
              if (err) return;
              setFile(f);
              setBbox(null);
            }}
            onBbox={setBbox}
          />
          <div className="form-row">
            <label className="field">
              <span className="field__label">vehicle_id</span>
              <input
                className="input input--mono"
                value={vehicleId}
                onChange={(e) => setVehicleId(e.target.value)}
                autoComplete="off"
              />
              <span className="field__hint">Необязательно: идентичность из разметки</span>
            </label>
            <label className="field">
              <span className="field__label">ГРЗ</span>
              <input
                className="input input--mono"
                value={plate}
                onChange={(e) => setPlateValue(e.target.value)}
                maxLength={16}
                autoComplete="off"
                spellCheck={false}
              />
              <span className="field__hint">Необязательно: номер из другого источника, не с кадра</span>
            </label>
          </div>
          {error && (
            <div className="alert alert--error" role="alert">
              <AlertTriangle aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}
        </>
      )}
    </Modal>
  );
}

export function ImportDialog({ onClose, onDone }: { onClose: () => void; onDone: (r: ImportReport) => void }) {
  const [csv, setCsv] = useState<File | null>(null);
  const [dir, setDir] = useState("images");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ImportReport | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!csv) return;
    setBusy(true);
    setError(null);
    try {
      const r = await importGallery(csv, dir.trim());
      setReport(r);
      onDone(r);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      narrow
      eyebrow="Массовая загрузка"
      title="Импорт разметки CSV"
      busy={busy}
      onClose={onClose}
      onSubmit={submit}
      foot={
        report ? (
          <button type="button" className="btn btn--primary" onClick={(e) => e.currentTarget.closest("dialog")?.close()}>
            Готово
          </button>
        ) : (
          <button type="submit" className="btn btn--primary" disabled={!csv || busy} aria-busy={busy}>
            {busy && <span className="spinner" aria-hidden="true" />}
            {busy ? "Импортируем…" : "Импортировать"}
          </button>
        )
      }
    >
      {report ? (
        <div className="import-report" role="status">
          <div className="import-report__nums">
            <div>
              <span className="import-report__value">{int(report.imported)}</span>
              <span className="muted">импортировано</span>
            </div>
            <div>
              <span className="import-report__value">{int(report.failed)}</span>
              <span className="muted">с ошибкой</span>
            </div>
          </div>
          {report.errors.length > 0 && (
            <details>
              <summary>Первые ошибки</summary>
              <ul className="import-report__errors">
                {report.errors.slice(0, 20).map((er) => (
                  <li key={er.imageId}>
                    <span className="mono">{er.imageId}</span> — {er.message}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ) : (
        <>
          <p className="muted">
            Формат датасета: <code>image_id,x,y,w,h[,vehicle_id,camera_id]</code>. Кадры сервис читает со своего
            диска — из каталога датасета, смонтированного в api-gateway, поэтому загружается только CSV.
          </p>
          <label className="field">
            <span className="field__label">CSV-файл *</span>
            <input
              className="input"
              type="file"
              accept=".csv,text/csv"
              required
              onChange={(e) => setCsv(e.target.files?.[0] ?? null)}
            />
          </label>
          <label className="field">
            <span className="field__label">Каталог кадров</span>
            <input className="input input--mono" value={dir} onChange={(e) => setDir(e.target.value)} />
            <span className="field__hint">Относительно DATASET_DIR сервиса; по умолчанию images</span>
          </label>
          {busy && (
            <p className="muted" role="status">
              Импорт синхронный: тысячи кадров — это минуты. Окно можно не закрывать, результат появится здесь.
            </p>
          )}
          {error && (
            <div className="alert alert--error" role="alert">
              <AlertTriangle aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}
        </>
      )}
    </Modal>
  );
}
