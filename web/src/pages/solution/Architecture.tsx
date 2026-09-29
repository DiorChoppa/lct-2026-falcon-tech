import type { CSSProperties } from "react";

/**
 * Схема контейнеров (C4, уровень 2) в HTML: ярусы сверху вниз, между ярусами —
 * протокол. Источник правды — docs/04-architecture.md; здесь фактическое
 * состояние compose: кропы на общем томе через object_store (S3 — целевой вариант).
 */
const TIERS: { proto?: string; nodes: { name: string; tech: string; role: string; kind?: "ext" | "db" | "opt" }[] }[] = [
  {
    nodes: [
      { name: "Оператор", tech: "браузер", role: "поиск, сверка деталей, галерея, экспорт", kind: "ext" },
      { name: "Жюри", tech: "docker run --gpus all", role: "три файла сдачи одной командой", kind: "ext" },
    ],
  },
  {
    proto: "HTTPS · REST/JSON",
    nodes: [
      { name: "web", tech: "nginx · React 19", role: "тонкий клиент, без бизнес-логики" },
      { name: "reid-submit", tech: "Rust · ONNX Runtime CUDA", role: "офлайн-экстрактор: без сети, БД и UI" },
    ],
  },
  {
    proto: "REST · OpenAPI 3 (utoipa)",
    nodes: [{ name: "api-gateway", tech: "Rust · axum", role: "приём кадра и bbox, валидация, Swagger, маршрутизация" }],
  },
  {
    proto: "gRPC · proto/reid/v1",
    nodes: [
      { name: "search", tech: "Rust · tonic", role: "kNN, порог, отказ, сверка патчей, экспорт" },
      { name: "gallery", tech: "Rust · sqlx", role: "записи, кропы, ГРЗ, импорт, миграции" },
      { name: "inference", tech: "Rust · ort", role: "кроп, препроцессинг, эмбеддинг, патч-токены" },
      { name: "tagger", tech: "Python", role: "zero-shot теги деталей, асинхронно", kind: "opt" },
    ],
  },
  {
    proto: "SQL · файловое хранилище",
    nodes: [
      { name: "PostgreSQL 16 + pgvector", tech: "HNSW · cosine", role: "галерея, векторы 1024-d, история поисков", kind: "db" },
      { name: "Том кропов", tech: "object_store", role: "кропы галереи и запросов; S3 — целевой", kind: "db" },
    ],
  },
];

export function Architecture() {
  return (
    <figure className="arch card" aria-labelledby="arch-cap">
      <figcaption id="arch-cap" className="chart__head">
        <span className="chart__title">Компоненты и границы ответственности</span>
        <span className="chart__sub">
          Каждый этап ТЗ — у одного сервиса-владельца; в Postgres у gallery и search свои роли: search_ro только читает галерею
        </span>
      </figcaption>
      <ol className="arch__tiers">
        {TIERS.map((tier, i) => (
          <li key={i} className="arch__tier">
            {tier.proto && (
              <div className="arch__proto">
                <span>{tier.proto}</span>
              </div>
            )}
            <ul className="arch__nodes" style={{ "--n": tier.nodes.length } as CSSProperties}>
              {tier.nodes.map((n) => (
                <li key={n.name} className={`arch__node${n.kind ? ` arch__node--${n.kind}` : ""}`}>
                  <span className="arch__name">{n.name}</span>
                  <span className="arch__tech mono">{n.tech}</span>
                  <span className="arch__role">{n.role}</span>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
      <p className="arch__foot muted">
        Логика инференса и поиска — в библиотечных крейтах, которые линкуют и сервисы, и CLI: пакетный прогон
        считается тем же кодом без сети. Пунктир — опциональный сервис: без него всё работает, просто без тегов.
      </p>
    </figure>
  );
}
