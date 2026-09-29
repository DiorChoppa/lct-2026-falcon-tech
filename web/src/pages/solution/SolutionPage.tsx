import {
  ArrowUpRight,
  Boxes,
  CheckCircle2,
  CircleDashed,
  Cpu,
  Crop,
  Database,
  FileCheck2,
  Gauge,
  Map as MapIcon,
  Route as RouteIcon,
  Scale,
  ScanSearch,
  ShieldCheck,
  Target,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import type { Info } from "../../api/types";
import { Link, ROUTES, routeHref } from "../../app/router";
import { usePageFocus } from "../../app/usePageFocus";
import thresholdData from "../../data/threshold.json";
import { dec } from "../../lib/format";
import { Architecture } from "./Architecture";
import { ConfidenceStrip, ScoreMeter, StackedBar, ThresholdChart } from "./charts";
import "./solution.css";
import "./charts.css";

const op = thresholdData.operatingPoint;
const audit = thresholdData.audit;

/** Критерии оценки ТЗ §9 — порядок и веса как у жюри. */
const CRITERIA = [
  { id: "accuracy", weight: 45, title: "Точность на закрытом тесте", short: "Точность", icon: Target },
  { id: "performance", weight: 20, title: "Производительность и применимость", short: "Скорость", icon: Gauge },
  { id: "engineering", weight: 15, title: "Инженерное качество", short: "Инженерия", icon: Wrench },
  { id: "refusal", weight: 10, title: "Корректность режима предложения кандидатов", short: "Отказ", icon: Scale },
  { id: "defense", weight: 10, title: "Защита решения", short: "Защита", icon: ShieldCheck },
] as const;

const TOC = [
  { id: "summary", label: "Коротко" },
  { id: "stages", label: "Этапы сервиса · §4" },
  ...CRITERIA.map((c) => ({ id: c.id, label: `${c.short} · ${c.weight}%` })),
  { id: "extras", label: "Доп. возможности · §10" },
  { id: "delivery", label: "Сдача и правила · §8" },
  { id: "product", label: "Развитие продукта" },
  { id: "limits", label: "Ограничения" },
];

export default function SolutionPage({ info }: { info: Info | null }) {
  const titleRef = usePageFocus<HTMLHeadingElement>("Решение по критериям ТЗ");
  const current = useScrollSpy(TOC.map((t) => t.id));

  return (
    <div className="page solution">
      <div className="solution__layout">
        <nav className="toc" aria-label="Разделы описания">
          <p className="toc__title">На странице</p>
          <ol>
            {TOC.map((t) => (
              <li key={t.id}>
                <a href={`#${t.id}`} aria-current={current === t.id ? "location" : undefined}>
                  {t.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <article className="doc">
          {/* ——— Коротко ——— */}
          <header id="summary" className="hero">
            <p className="eyebrow">Кейс Street Falcon · ЛЦТ 2026</p>
            <h1 ref={titleRef} tabIndex={-1}>
              Цифровой признак автомобиля без госномера — решение по критериям ТЗ
            </h1>
            <p className="hero__lead">
              Сервис строит по кропу ТС вектор внешнего вида (1024 числа) и ищет тот же автомобиль на снимках других
              камер. Возвращает топ-N кандидатов с уверенностью либо честный отказ. No OCR or plate text enters the encoder. Image appearance may still contain plate-region information; the masking control below measures sensitivity.
            </p>

            <dl className="kpis">
              <Kpi label="mAP@10" value="0,870" note="отложенный аудит: 200 запросов, 20 % без пары" />
              <Kpi label="Rank-1" value="87,5 %" note="аудит: первый кандидат — та же машина, кросс-камерно" />
              <Kpi label="Задержка batch=1" value="22,42 мс" note="full extraction, local RTX 3090; A5000 untested" />
              <Kpi label="Пропускная способность" value="181,55 FPS" note="batch 32, тот же стенд" />
              <Kpi label="Режим отказа" value={`F1 ${dec(audit.f1, 2)}`} note={`live cosine audit, TNR ${dec(audit.tnr, 3)}; calibration threshold`} />
              <Kpi label="Веса модели" value="607 МБ" note="лимит ТЗ — 2 ГБ" />
            </dl>

            <div className="scorebar" aria-label="Веса критериев оценки">
              <p className="scorebar__title">Итоговая оценка по ТЗ §9 — каждый блок ведёт к своему разделу</p>
              <ol className="scorebar__track">
                {CRITERIA.map(({ id, weight, short, icon: Icon }) => (
                  <li key={id} style={{ flexGrow: weight }}>
                    <a href={`#${id}`} className="scorebar__seg">
                      <span className="scorebar__weight">{weight}%</span>
                      <span className="scorebar__name">
                        <Icon aria-hidden="true" />
                        {short}
                      </span>
                    </a>
                  </li>
                ))}
              </ol>
            </div>

            <p className="hero__caveat">
              <strong>Evidence scope.</strong> Final native SEARCH mAP@10 is 92.005%; audit is 87.003% on 160 known and 40 unknown queries. SEARCH guided development. The audit fold was already opened for the previous model; it is not a pristine holdout. Latency and throughput are measured locally on RTX 3090, not on the jury A5000 or driver 535.</p>
          </header>

          {/* ——— §4 ——— */}
          <Section id="stages" eyebrow="Функциональные требования · ТЗ §4" title="Четыре этапа — четыре владельца">
            <p className="doc__p">
              ТЗ делит работу сервиса на последовательные этапы. У каждого — один сервис-владелец, поэтому модель можно
              менять, не трогая логику порога, а порог — не переобучая модель.
            </p>
            <ol className="stages">
              <Stage
                n={1}
                icon={Crop}
                title="Получение"
                owner="api-gateway"
                text="Кадр JPEG/PNG до 3 МиБ и bbox через REST или веб-интерфейс. Валидация: формат, размер, рамка в пределах кадра (выход на 1–2 px из разметки обрезается), понятные ошибки 422."
              />
              <Stage
                n={2}
                icon={Cpu}
                title="Обработка"
                owner="inference"
                text="Кроп строго по bbox, растяжение до 256×256 (bicubic), нормализация ImageNet — всё по манифесту models/model.json. ONNX Runtime → вектор float32 1024-d с L2-нормой."
              />
              <Stage
                n={3}
                icon={Database}
                title="Анализ"
                owner="search"
                text="Косинусное сходство с галереей в PostgreSQL + pgvector (HNSW-индекс). Векторы одной машины с разных камер ближе, чем векторы разных машин, — этому учит метрическое обучение."
              />
              <Stage
                n={4}
                icon={ScanSearch}
                title="Результат"
                owner="search"
                text={`Сортировка по убыванию уверенности, топ-N с пометкой «принят / ниже порога», отказ, если выше порога ${dec(thresholdData.threshold)} никого нет. Экспорт в CSV, история поисков в БД.`}
              />
            </ol>
            <p className="doc__note">
              Дополнительные атрибуты ТС (ТЗ допускает их опционально) — асинхронный сервис <code>tagger</code> с
              zero-shot тегами деталей. Проводка и тесты готовы, но веса детектора в поставку не входят (
              <code>NullBackend</code>), поэтому на поиск он сейчас не влияет.
            </p>
          </Section>

          {/* ——— 45% ——— */}
          <Criterion id="accuracy" requirement="mAP по кросс-камерным запросам на скрытой части теста; Rank-1 и Rank-5 — дополнительно. Совпадения с той же камеры исключаются." status="done" statusText="mAP@10 0,870 на отложенном аудите">
            <div className="cols">
              <div>
                <h3 className="doc__h3">Модель</h3>
                <ul className="doc__list">
                  <li>
                    <strong>Backbone — DINOv3 ViT-L/16</strong> (Meta AI, self-supervised). Выбран по измерениям на
                    валидации среди DINOv2, CLIP и DINOv3; большая модель укладывается в бюджет скорости жюри с запасом, а
                    её патч-токены дают объяснимость без второй модели.
                  </li>
                  <li>
                    <strong>Training lineage.</strong> Public VERI-Wild TRAIN E2, then TRAIN+public TEST E1 (416,314 images, 40,671 identities; Q46 allows external test splits), then eight epochs on 5,722 FIT-v8 crops / 928 identities. The 24 historical FIT exclusions are disclosed. Fresh FIT retraining scored 91.213% SEARCH.</li>
                  <li>
                    <strong>Ракурс и освещение.</strong> Сэмплер «2+2»: у каждой машины в батче кадры с двух разных
                    камер, поэтому положительные пары всегда кросс-камерные — модель учит то, что не меняется при смене
                    точки съёмки. Голова BNNeck, L2-нормированный выход.
                  </li>
                  <li>
                    <strong>Open-set.</strong> Классов на выходе нет: тестовые машины сравниваются по расстоянию, модель их
                    никогда не видела.
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="doc__h3">Валидация повторяет протокол теста</h3>
                <ul className="doc__list">
                  <li>Сплит по машинам, а не по кадрам; одна камера — в галерею, остальные — в запросы.</li>
                  <li>
                    Junk-фильтр как у жюри: та же машина <em>и</em> та же камера убирается до усечения до 10.
                  </li>
                  <li>Часть машин без пары в галерее — чтобы мерить отказ (в закрытом тесте их 20 %).</li>
                  <li>
                    Метрики считает неизменённый <code>evaluate.py</code> организаторов; наш <code>eval.py</code> сверен с
                    ним тестом паритета.
                  </li>
                </ul>
              </div>
            </div>
            <Table
              caption="Final native CUDA model: DBA and frozen calibration threshold"
              head={["Episode", "mAP@10", "Rank-1", "TP / FP / FN / TN"]}
              rows={[
                ["SEARCH development", "0.92005", "0.91279", "148 / 0 / 24 / 43"],
                ["Calibration", "0.83000", "0.79375", "142 / 0 / 18 / 40"],
                ["Previously opened audit", "0.87003", "0.87500", "141 / 2 / 19 / 38"],
              ]}
            />
            <p className="doc__note">
              Для масштаба: без дообучения лучший из проверенных backbone (CLIP ViT-B/16) давал mAP 0,162 на нашей
              валидации, DINOv2 ViT-B/14 — 0,118. Протоколы разные, сравнивать напрямую нельзя, но порядок роста виден.
            </p>
          </Criterion>

          {/* ——— 20% ——— */}
          <Criterion id="performance" requirement="Время признака на одно ТС при batch=1 и устойчивый FPS в батче на GPU жюри (RTX A5000); веса ≤ 2 ГБ, без OOM. Баллы: ≤ 40 мс и ≥ 100 FPS — полные." status="done" statusText="в зоне полного балла">
            <div className="meters">
              <ScoreMeter
                label="Задержка, batch = 1"
                value={22.42}
                unit="мс"
                max={100}
                zones={[40, 80]}
                lowerIsBetter
                note="медиана, p95 30,34 мс; в таймере весь путь: чтение файла → JPEG-декод → кроп → препроцессинг → forward → L2"
              />
              <ScoreMeter
                label="Пропускная способность"
                value={181.55}
                unit="FPS"
                max={250}
                zones={[50, 100]}
                lowerIsBetter={false}
                note="лучший устойчивый результат при batch 32 (из 1 / 8 / 16 / 32)"
              />
              <ScoreMeter
                label="Суммарный размер весов"
                value={607}
                unit="МБ"
                max={2048}
                zones={[2048, 2048]}
                lowerIsBetter
                note="один ONNX-граф смешанной точности FP16/FP32, 607 325 498 bytes; model version 1.2.0-final"
              />
            </div>
            <div className="cols">
              <div>
                <h3 className="doc__h3">Как достигнуто</h3>
                <ul className="doc__list">
                  <li>
                    Нативный экстрактор на Rust: turbojpeg-декод, ресайз бит-в-бит как в Pillow, ONNX Runtime CUDA, без
                    Python на горячем пути.
                  </li>
                  <li>Декод и кроп следующих кадров идут с предзагрузкой — GPU не простаивает в батче.</li>
                  <li>Одна модель на горячем пути: любые детали и переранжирование — только поверх топ-K.</li>
                </ul>
              </div>
              <div>
                <h3 className="doc__h3">Честно о цифрах</h3>
                <ul className="doc__list">
                  <li>
                    Замер — на RTX 3090, не на A5000 жюри; официальные баллы даст только скрипт организаторов.
                  </li>
                  <li>
                    Historical demo timing was CPU-only. The final jury image is independently measured on CUDA. Public demo availability is being restored by the team; its current latency is not claimed.
                  </li>
                </ul>
              </div>
            </div>
          </Criterion>

          {/* ——— 15% ——— */}
          <Criterion id="engineering" requirement="Воспроизводимость из исходников, чистая архитектура, Docker, стек Python / C++ / Rust, OpenAPI, полная документация." status="done" statusText="docker compose up — одной командой">
            <Architecture />
            <div className="facts">
              <Fact icon={Boxes} title="Запуск одной командой">
                <code>docker compose up --build</code> поднимает db, inference, gallery, search, tagger, api-gateway и
                web. Веса лежат в Git LFS и запекаются в образы — интернет в рантайме не нужен. GPU — override-файлом.
              </Fact>
              <Fact icon={FileCheck2} title="Контракты из кода">
                OpenAPI генерируется из Rust-кода (utoipa), Swagger — на <code>/api/docs</code>. gRPC — единый источник{" "}
                <code>proto/reid/v1</code>. Препроцессинг — манифест <code>models/model.json</code> рядом с весами.
              </Fact>
              <Fact icon={CheckCircle2} title="Паритет трёх путей инференса">
                Independent FP32 PyTorch versus Linux CUDA cosine agreement is at least 0.99999717 on FIT fixtures. Reversing queries or extracting a singleton preserves all tested rankings and refusals. Full native output files repeat byte-for-byte.
              </Fact>
              <Fact icon={Wrench} title="Тесты и CI">
                Rust-тесты без БД и сети (фейковые gRPC-клиенты, моки репозиториев), pytest метрик, vitest веба. CI: fmt,
                clippy, cargo test, ruff, сборка web, автодеплой демо-стенда с HTTPS.
              </Fact>
            </div>
            <p className="doc__note">
              Стек: Rust 1.92 (axum, tonic, sqlx, ort), PostgreSQL 16 + pgvector, React 19 + Vite. Python — только
              обучение и опциональный <code>tagger</code>. Документация: <code>README.md</code>, концепция, архитектура,
              данные, масштабирование, журнал решений (ADR) и ответы организаторов в <code>docs/</code>.
            </p>
          </Criterion>

          {/* ——— 10% отказ ——— */}
          <Criterion id="refusal" requirement="Сервис аргументированно возвращает пустой ответ, если уверенного совпадения нет. Оценка — F1 и TNR при пороге команды, порог обосновывается на защите." status="done" statusText={`F1 ${dec(op.f1)} · TNR ${dec(op.tnr, 2)}`}>
            <div className="formula card">
              <div>
                <span className="formula__label">Правило</span>
                <p>
                  Ответ, если уверенность топ-1 ≥ <strong>{dec(thresholdData.threshold)}</strong>, иначе отказ.
                  Уверенность — косинусное сходство L2-нормированных векторов.
                </p>
              </div>
              <div>
                <span className="formula__label">Как выбран порог</span>
                <p>
                  exact observed boundary maximizing calibration Q <strong>0,7 · F1 + 0,3 · TNR</strong> на эпизоде калибровки
                  (200 запросов, 40 без пары; model frozen before this calibration; fold previously used for the incumbent) и один раз проверен на
                  аудите. F1 и TNR — на уровне запроса, по верхнему кандидату, как в <code>evaluate.py</code>.
                </p>
              </div>
              <div className="cm" role="table" aria-label="Матрица ошибок при выбранном пороге">
                <div role="row" className="cm__row">
                  <span role="cell" className="cm__cell cm__cell--good">
                    <strong>{op.tp}</strong> TP<small>верный ответ</small>
                  </span>
                  <span role="cell" className="cm__cell cm__cell--good">
                    <strong>{op.tn}</strong> TN<small>верный отказ</small>
                  </span>
                </div>
                <div role="row" className="cm__row">
                  <span role="cell" className="cm__cell cm__cell--bad">
                    <strong>{op.fp}</strong> FP<small>ложный ответ</small>
                  </span>
                  <span role="cell" className="cm__cell cm__cell--warn">
                    <strong>{op.fn}</strong> FN<small>лишний отказ</small>
                  </span>
                </div>
              </div>
            </div>
            <ThresholdChart />
            <ConfidenceStrip />
            <div className="cols">
              <div>
                <h3 className="doc__h3">Почему именно {dec(thresholdData.threshold)}</h3>
                <ul className="doc__list">
                  <li>
                    На калибровке это вершина балла жюри: {dec(op.score)}, {op.fp} false positive answers. На отложенном
                    аудите тот же порог даёт {dec(audit.score)}: {audit.tp} TP, {audit.fp} FP, {audit.fn} FN,{" "}
                    {audit.tn} TN.
                  </li>
                  <li>
                    The threshold maximizes calibration Q across every observed boundary, including accept-all and reject-all decisions. Audit results do not change it.
                  </li>
                  <li>
                    Refusal can reject a genuine match. The operator still sees the ranked candidates with a below-threshold label.
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="doc__h3">Что мы не прячем</h3>
                <ul className="doc__list">
                  <li>
                    The hidden test may have a different score distribution. No specific hidden-test TNR is guaranteed; the separate audit figures above show the observed outcome.
                  </li>
                  <li>
                    Здесь — порог живого сервиса (косинус без DBA). Файлы сдачи ранжируют после DBA по галерее и держат
                    свой порог 0,720701, выбранный тем же способом; шкалы разные, пороги не взаимозаменяемы.
                  </li>
                  <li>
                    В <code>candidates.csv</code> отказ — отсутствие строк по запросу; формат подтверждён организаторами.
                  </li>
                </ul>
              </div>
            </div>
          </Criterion>

          {/* ——— 10% защита ——— */}
          <Criterion id="defense" requirement="Обоснование архитектуры, глубина анализа ошибок, защита порога уверенности." status="done" statusText="разбор ошибок по каждому запросу">
            <p className="doc__p">
              Разобраны все 215 запросов эпизода: ранжирование, отказ и причины ошибок. Среди просмотренных глазами
              случаев чаще всего модель подводят смена точки съёмки и машины-двойники одной модели и цвета.
            </p>
            <div className="cols cols--charts">
              <StackedBar title="SEARCH known queries" sub="172 queries after same-camera filtering" total={172}
                parts={[
                  { label: "Correct first result", value: 157, tone: "ok" },
                  { label: "Wrong first result", value: 15, tone: "bad" },
                ]} />
            </div>
            <div className="cases">
              <div className="case card">
                <p className="eyebrow">SEARCH ID 1258</p>
                <p>
                  For this verified TT SEARCH case, the first correct front view ranks ninth after same-camera filtering; only one of two valid views appears in the top ten. Similar rear-view vehicles outrank it. Labels and thresholds were not changed for this case.
                </p>
              </div>
              <div className="case card">
                <p className="eyebrow">Что дальше</p>
                <p>
                  Устойчивость к контексту кропа и противоположным ракурсам — гипотезы только на train, без подгонки
                  под эпизод. Патч-сверка уже показывает оператору, когда «общих деталей мало».
                </p>
              </div>
              <Link href={ROUTES.search} className="case case--link card">
                <p className="eyebrow">Живое демо</p>
                <p>
                  Четыре сценария на экране поиска: уверенное совпадение, случай у порога, честный отказ и ошибка из-за
                  ракурса.
                </p>
                <span className="case__go">
                  Открыть поиск <ArrowUpRight aria-hidden="true" />
                </span>
              </Link>
            </div>
          </Criterion>

          {/* ——— §10 ——— */}
          <Section id="extras" eyebrow="Дополнительные возможности · ТЗ §10" title="Тай-брейкеры: что уже работает">
            <div className="extras">
              <Extra status="done" title="Демонстрационный веб-интерфейс">
                Этот интерфейс: загрузка кадра и рамки, топ-N с уверенностью и отказом, сверка деталей, экспорт CSV,
                галерея с привязкой ГРЗ. Стенд с HTTPS, API описан в Swagger.
              </Extra>
              <Extra status="partial" title="Интерпретируемость">
                «Сверить детали» — взаимно ближайшие патч-токены 16×16 того же ViT: на обоих снимках подсвечены
                совпавшие области (логотип, QR-код, ручка двери). Grad-CAM и attention-карты для глобального вектора — в плане.
              </Extra>
              <Extra status="done" title="Масштабируемость: ANN на 10⁶">
                pgvector HNSW на 10⁶ векторов (512-d, один узел): p50 33 мс против 1 733 мс полным перебором —
                ≈ 50 × быстрее, recall@10 0,79–0,85 в зависимости от ef_search. Шардинг и квантизация — в плане
                масштабирования.
              </Extra>
              <Extra status="done" title="Глубокий анализ ошибок">
                По каждому из 215 запросов: ранги позитивов, junk, категории отказа, визуальный разбор с
                описанием причин — раздел «Защита» выше.
              </Extra>
            </div>
            <Table
              caption="ANN-бенчмарк: pgvector 0.8.6, 10⁶ синтетических векторов 512-d, Docker на M4 Pro (bench/results)"
              head={["Конфигурация", "Поиск p50 / p95", "recall@10", "Индекс"]}
              rows={[
                ["Полный перебор", "1 733 мс / 3 219 мс", "1,00", "—"],
                ["HNSW m=24, ef_c=128, ef_search=100", "33 мс / 60 мс", "0,79", "1 669 с, 2,5 ГБ"],
                ["то же, ef_search=400", "44 мс / 89 мс", "0,85", ""],
                ["10⁵ векторов, m=16, ef_c=64", "4,4 мс", "0,92", "81 с"],
              ]}
            />
          </Section>

          {/* ——— §8 и правила ——— */}
          <Section id="delivery" eyebrow="Формат сдачи · ТЗ §8 · правила" title="Артефакты и запрет на номер">
            <div className="facts">
              <Fact icon={FileCheck2} title="Три файла одной командой">
                <code>submission.csv</code> (топ-10 на запрос), <code>embeddings.npy</code> (в порядке CSV),{" "}
                <code>candidates.csv</code> (принятые с уверенностью или отказ). Образ <code>reid-submit</code>: x86_64 +
                CUDA 12, без сети при запуске.
              </Fact>
              <Fact icon={ShieldCheck} title="ГРЗ не используется">
                No camera, time, filename or OCR enters the model. Approximate plate masking on SEARCH changes mAP from 92.005% to 91.292%; a same-area random control scores 91.802%. This does not prove zero plate dependence and is not the exact jury mask test.
              </Fact>
              <Fact icon={Database} title="Внешние ресурсы открыты">
                Веса DINOv3 (Meta, DINOv3 License), датасет VERI-Wild (некоммерческое исследовательское использование)
                — только для обучения. Пересечение с данными конкурса проверено по хэшам. Полный список с версиями — в
                README.
              </Fact>
            </div>
            <pre className="code card">
              <code>{`docker run --rm --network none --gpus all -v /data:/data -v $PWD/submission:/out reid-submit \\
    --images /data/images --query /data/test_query.csv \\
    --gallery /data/test_gallery.csv --out /out`}</code>
            </pre>
          </Section>

          {/* ——— Развитие продукта ——— */}
          <Section id="product" eyebrow="Развитие продукта" title="От кадра — к маршруту и местам стоянок">
            <p className="doc__p">
              Когда у кадров есть камера и время, найденная машина превращается в маршрут: где и когда она появлялась по
              Москве и в поездках, где регулярно стоит ночью и днём. В сдаче этого нет: по ТЗ (§5.2) задача решается
              только по визуальным признакам, и ответ поиска, candidates.csv и метрика от маршрута не зависят.
            </p>
            <div className="facts">
              <Fact icon={MapIcon} title="Как устроено в продукте">
                Реестр камер с координатами; таблица появлений (камера, время, эмбеддинг, идентичность). Появления
                связываются в идентичность по ReID и подтверждаются проверкой скорости между камерами; места стоянок и
                гипотезы считает фоновая задача. Ложится на существующие gallery и search без новой модели.
              </Fact>
              <Fact icon={ShieldCheck} title="Как это усилит ReID в продукте">
                Двойники одной модели и цвета — вторая по частоте причина ошибок в разборе (3 из 18, после смены
                ракурса). Похожая машина в 40 км через 9 минут физически не наша: проверка отсекает её, поэтому
                кандидатов чуть ниже порога, вероятно, можно брать без роста ложных ответов. Это гипотеза: проверить
                её можно только на данных с камерой и временем, а на закрытом тесте их нет — в сдаче этого нет.
              </Fact>
              <Fact icon={Database} title="Данные уже есть у заказчика">
                Парковочные комплексы Street Falcon фиксируют стоящую машину с временем и местом — это готовый источник
                для «мест стоянок» и длительности стоянки.
              </Fact>
              <Fact icon={Scale} title="Правовые рамки">
                Траектория с привязанным ГРЗ — персональные данные (152-ФЗ). Места — факт «регулярная ночная / дневная
                стоянка», интерпретация — только гипотеза с уверенностью для оператора. Доступ по ролям, журнал аудита
                каждого просмотра, срок хранения, законное основание запроса.
              </Fact>
            </div>
            <Link href={routeHref("959")} className="btn btn--primary btn--lg product__cta">
              <RouteIcon aria-hidden="true" />
              Открыть пример маршрута
            </Link>
            <p className="doc__note">
              Пример — на смоделированных камерах и времени: в данных конкурса их нет, а анонимный camera_id датасета не
              используем (ответ жюри на вопрос о camera_id). Реальны кадры машины, найденные моделью, и их уверенность;
              двойника симулятор ставит так, чтобы показать механизм проверки.
            </p>
          </Section>

          {/* ——— Ограничения ——— */}
          <Section id="limits" eyebrow="Честно" title="Ограничения и следующие шаги">
            <ul className="limits">
              <li>
                <CircleDashed aria-hidden="true" />
                <span>Аудит — 200 запросов: 95 % интервал mAP@10 от 0,823 до 0,911. Камеры и машины закрытого теста другие,
                реальная цифра может выйти в любую сторону этого интервала.</span>
              </li>
              <li>
                <CircleDashed aria-hidden="true" />
                <span>Запас по TNR мал: на аудите 2 из 40 запросов без пары прошли порог (TNR 0,95).</span>
              </li>
              <li>
                <CircleDashed aria-hidden="true" />
                <span>The public demo is being restored; final jury CUDA timing is reported separately.</span>
              </li>
              <li>
                <CircleDashed aria-hidden="true" />
                <span>Теги деталей: проводка есть, весов детектора в поставке нет. Grad-CAM — в плане.</span>
              </li>
              <li>
                <CircleDashed aria-hidden="true" />
                <span>Масштаб города: шина событий, Triton с dynamic batching и векторная БД с партициями по времени и району —
                план в <code>docs/05-scaling.md</code>; контракты сервисов при этом не меняются.</span>
              </li>
            </ul>
            {info?.model && (
              <p className="live mono">
                Сейчас в сервисе: {info.model.name} {info.model.version} · {info.model.dim}-d · вход{" "}
                {info.model.inputWidth}×{info.model.inputHeight} · {info.model.executionProvider}
              </p>
            )}
          </Section>
        </article>
      </div>
    </div>
  );
}

/** Какой раздел сейчас в верхней части экрана — для подсветки в оглавлении. */
function useScrollSpy(ids: string[]): string {
  const [current, setCurrent] = useState(ids[0]);
  useEffect(() => {
    const els = ids.map((id) => document.getElementById(id)).filter((e): e is HTMLElement => !!e);
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setCurrent(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -65% 0px" },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [ids.join()]);
  return current;
}

function Kpi({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="kpi">
      <dt>{label}</dt>
      <dd>
        <span className="kpi__value">{value}</span>
        <span className="kpi__note">{note}</span>
      </dd>
    </div>
  );
}

function Section(props: { id: string; eyebrow: string; title: string; children: ReactNode }) {
  return (
    <section id={props.id} className="doc__section" aria-labelledby={`${props.id}-title`}>
      <p className="eyebrow">{props.eyebrow}</p>
      <h2 id={`${props.id}-title`}>{props.title}</h2>
      {props.children}
    </section>
  );
}

function Criterion(props: {
  id: (typeof CRITERIA)[number]["id"];
  requirement: string;
  status: "done" | "partial";
  statusText: string;
  children: ReactNode;
}) {
  const c = CRITERIA.find((x) => x.id === props.id)!;
  const Icon = c.icon;
  return (
    <section id={c.id} className="doc__section criterion" aria-labelledby={`${c.id}-title`}>
      <div className="criterion__head">
        <span className="criterion__weight" aria-label={`Вес критерия ${c.weight} процентов`}>
          {c.weight}
          <small>%</small>
        </span>
        <div>
          <p className="eyebrow">
            <Icon aria-hidden="true" /> Критерий ТЗ §9
          </p>
          <h2 id={`${c.id}-title`}>{c.title}</h2>
        </div>
        <span className={`badge ${props.status === "done" ? "badge--ok" : "badge--warn"} criterion__status`}>
          {props.status === "done" ? <CheckCircle2 aria-hidden="true" /> : <CircleDashed aria-hidden="true" />}
          {props.statusText}
        </span>
      </div>
      <blockquote className="req">
        <span className="req__label">Требование</span>
        {props.requirement}
      </blockquote>
      {props.children}
    </section>
  );
}

function Stage(props: { n: number; icon: LucideIcon; title: string; owner: string; text: string }) {
  const Icon = props.icon;
  return (
    <li className="stage card">
      <span className="stage__n mono">0{props.n}</span>
      <span className="stage__icon" aria-hidden="true">
        <Icon />
      </span>
      <h3>{props.title}</h3>
      <span className="badge badge--brand mono">{props.owner}</span>
      <p>{props.text}</p>
    </li>
  );
}

function Fact(props: { icon: LucideIcon; title: string; children: ReactNode }) {
  const Icon = props.icon;
  return (
    <div className="fact card">
      <span className="fact__icon" aria-hidden="true">
        <Icon />
      </span>
      <h3>{props.title}</h3>
      <p>{props.children}</p>
    </div>
  );
}

function Extra(props: { status: "done" | "partial"; title: string; children: ReactNode }) {
  return (
    <div className={`extra card extra--${props.status}`}>
      <span className={`badge ${props.status === "done" ? "badge--ok" : "badge--warn"}`}>
        {props.status === "done" ? <CheckCircle2 aria-hidden="true" /> : <CircleDashed aria-hidden="true" />}
        {props.status === "done" ? "есть" : "частично"}
      </span>
      <h3>{props.title}</h3>
      <p>{props.children}</p>
    </div>
  );
}

function Table(props: { caption: string; head: string[]; rows: string[][] }) {
  return (
    <figure className="dtable card">
      <figcaption>{props.caption}</figcaption>
      <div className="dtable__scroll">
        <table>
          <thead>
            <tr>
              {props.head.map((h) => (
                <th key={h} scope="col">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {props.rows.map((r) => (
              <tr key={r[0]}>
                {r.map((c, i) => (i === 0 ? <th key={i} scope="row">{c}</th> : <td key={i}>{c}</td>))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
