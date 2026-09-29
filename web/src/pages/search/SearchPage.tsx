import { AlertTriangle, ArrowRight, Crop, Database, Scale, ScanSearch, Sparkles } from "lucide-react";
import { useEffect } from "react";
import type { Info } from "../../api/types";
import { usePageFocus } from "../../app/usePageFocus";
import { FramePicker } from "../../components/FramePicker";
import { Results } from "../../components/Results";
import { isUsable } from "../../lib/bbox";
import { dec } from "../../lib/format";
// Именованный импорт JSON — Vite оставит в чанке только число, не всю кривую.
import { threshold as modelThreshold } from "../../data/threshold.json";
import { SCENARIOS } from "./scenarios";
import type { SearchSession } from "./useSearchSession";
import "./search.css";

export function SearchPage({ session, info }: { session: SearchSession; info: Info | null }) {
  const titleRef = usePageFocus<HTMLHeadingElement>("Поиск ТС");
  const { file, bbox, state } = session;
  const step = !file ? 1 : !isUsable(bbox) ? 2 : 3;

  // Кадр из буфера обмена: скриншот с камеры вставляется без сохранения в файл.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith("image/"));
      if (!f) return;
      e.preventDefault();
      session.pickFile(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [session.pickFile]);

  const input = { w: info?.model?.inputWidth ?? 256, h: info?.model?.inputHeight ?? 256 };

  return (
    <div className="page search">
      <div className="page__head">
        <div>
          <p className="eyebrow">Оператор · поиск по внешнему виду</p>
          <h1 ref={titleRef} tabIndex={-1}>
            Найти автомобиль в галерее
          </h1>
        </div>
        <p className="page__lead">
          Загрузите кадр с камеры и обведите машину. Сервис построит цифровой признак по её виду — форма, цвет,
          ливрея, детали — и найдёт тот же автомобиль на снимках других камер. Номер не используется.
        </p>
      </div>

      <div className="search__grid">
        <section className="query card" aria-labelledby="query-title">
          <header className="query__head">
            <h2 id="query-title" className="query__title">
              Запрос
            </h2>
            <ol className="steps" aria-label="Шаги">
              {["Кадр", "Рамка", "Поиск"].map((s, i) => (
                <li
                  key={s}
                  className={`steps__item${step > i + 1 ? " is-done" : ""}${step === i + 1 ? " is-current" : ""}`}
                  aria-current={step === i + 1 ? "step" : undefined}
                >
                  <span className="steps__num">{i + 1}</span>
                  {s}
                </li>
              ))}
            </ol>
          </header>

          <FramePicker
            fileUrl={session.fileUrl}
            fileName={file?.name}
            fileError={session.fileError}
            bbox={bbox}
            onFile={session.pickFile}
            onBbox={session.setBbox}
          />

          <div className="query__run">
            <button
              type="button"
              className="btn btn--primary btn--lg query__btn"
              onClick={session.run}
              disabled={!session.canRun}
              aria-busy={state.kind === "loading"}
            >
              {state.kind === "loading" ? <span className="spinner" aria-hidden="true" /> : <ScanSearch aria-hidden="true" />}
              {state.kind === "loading" ? "Ищем в галерее…" : "Найти в галерее"}
            </button>
            <span className="query__why muted" aria-live="polite">
              {!file ? "Сначала кадр" : !isUsable(bbox) ? "Обведите машину на кадре" : "Топ-10 кандидатов с уверенностью"}
            </span>
          </div>

          <section className="scenarios" aria-labelledby="scen-title">
            <h3 id="scen-title" className="scenarios__title">
              <Sparkles aria-hidden="true" /> Сценарии демо — кадр и рамка в один клик
            </h3>
            <ul className="scenarios__list">
              {SCENARIOS.map((s, i) => (
                <li key={s.id}>
                  <button
                    type="button"
                    className={`scenario${session.scenario?.id === s.id ? " is-active" : ""}`}
                    onClick={() => session.loadScenario(s)}
                    aria-pressed={session.scenario?.id === s.id}
                  >
                    <img src={s.thumb} alt="" loading="lazy" width={72} height={54} />
                    <span className="scenario__text">
                      <span className="scenario__title">
                        <span className="mono">{i + 1}</span> {s.title}
                      </span>
                      <span className="scenario__expect">{s.expect}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </section>

        <section className="answer" aria-labelledby="answer-title" aria-busy={state.kind === "loading"}>
          <h2 id="answer-title" className="visually-hidden">
            Результат
          </h2>
          {state.kind === "idle" && <HowItWorks />}
          {state.kind === "loading" && <LoadingResults />}
          {state.kind === "error" && (
            <div className="alert alert--error" role="alert">
              <AlertTriangle aria-hidden="true" />
              <div>
                <strong>Поиск не выполнен.</strong> {state.message}
                <div className="alert__actions">
                  <button type="button" className="btn btn--sm" onClick={session.run} disabled={!session.canRun}>
                    Повторить
                  </button>
                </div>
              </div>
            </div>
          )}
          {state.kind === "done" && (
            <Results result={state.result} ms={state.ms} stale={state.stale} onRerun={session.run} inputSize={input} />
          )}
        </section>
      </div>
    </div>
  );
}

function LoadingResults() {
  return (
    <div className="results" role="status">
      <div className="verdict card verdict--loading">
        <div className="verdict__query reticle is-scanning">
          <div className="skeleton" style={{ position: "absolute", inset: 0 }} />
        </div>
        <div className="verdict__body">
          <p className="verdict__title">Строим эмбеддинг и ищем ближайших…</p>
          <p className="muted">Кроп по рамке → ViT-L/16 → вектор 1024-d → kNN в pgvector → порог.</p>
        </div>
      </div>
      <ol className="cands" aria-hidden="true">
        {Array.from({ length: 6 }, (_, i) => (
          <li key={i} className="cand card">
            <div className="skeleton" style={{ aspectRatio: "4 / 3" }} />
            <div className="cand__body">
              <div className="skeleton" style={{ height: 28, width: "50%" }} />
              <div className="skeleton" style={{ height: 6 }} />
              <div className="skeleton" style={{ height: 48 }} />
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

const PIPELINE = [
  { icon: Crop, title: "Кроп по bbox", text: "Кадр и рамка проходят валидацию; модель видит только машину." },
  { icon: Sparkles, title: "Цифровой признак", text: "DINOv3 ViT-L/16, дообученный на ReID, → вектор 1024-d с L2-нормой." },
  { icon: Database, title: "Поиск в галерее", text: "Косинусное сходство, kNN в PostgreSQL + pgvector (HNSW)." },
  { icon: Scale, title: "Порог или отказ", text: "Выше порога — кандидат; ниже у всех — честный отказ." },
];

function HowItWorks() {
  return (
    <div className="how card">
      <p className="eyebrow">Как это работает</p>
      <h3 className="how__title">От кадра до решения — около 0,7 с на CPU сервера</h3>
      <ol className="how__steps">
        {PIPELINE.map(({ icon: Icon, title, text }, i) => (
          <li key={title} className="how__step" style={{ animationDelay: `${i * 60}ms` }}>
            <span className="how__icon" aria-hidden="true">
              <Icon />
            </span>
            <span className="how__name">{title}</span>
            <span className="how__text">{text}</span>
            {i < PIPELINE.length - 1 && <ArrowRight className="how__arrow" aria-hidden="true" />}
          </li>
        ))}
      </ol>
      <p className="how__foot muted">
        Порог {dec(modelThreshold)} выбран по баллу жюри 0,7·F1 + 0,3·TNR на development-эпизоде (215 запросов),
        до финальной калибровки — подробности и кривая на странице «Решение по ТЗ». Начните с готового сценария
        или загрузите свой кадр. Время на таймере результата включает загрузку кадра по сети.
      </p>
    </div>
  );
}
