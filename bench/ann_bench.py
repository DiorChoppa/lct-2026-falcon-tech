# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "psycopg[binary]>=3.2", "pgvector>=0.3"]
# ///
"""ANN-демо: галерея из N случайных векторов в pgvector, HNSW, латентность и recall.

Числа для docs/05-scaling.md §5 и тай-брейкера ТЗ §10 («ANN для галереи ~10^6»).
Синтетика повторяет структуру галереи: «машины» — случайные центры на сфере, кадры —
центр плюс шум (косинус к центру ≈ 1/sqrt(1+s²), при s=0.5 это ≈ 0.89; между машинами
≈ 0). Запросы — новые кадры существующих машин. Равномерно случайные векторы без
кластеров дают HNSW бессмысленно низкий recall и не похожи на эмбеддинги.

    just ann-bench                      # 10^6, dim из models/model.json
    uv run bench/ann_bench.py --n 100000 --keep

Отдельная таблица ann_bench, галерея сервиса не трогается. DATABASE_URL — как у api.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 20_000


def unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def views(rng: np.random.Generator, centres: np.ndarray, noise: float) -> np.ndarray:
    """Кадры машин: центр + гауссов шум с нормой ≈ noise, снова на сферу."""
    e = rng.standard_normal(centres.shape, dtype=np.float32) * (noise / np.sqrt(centres.shape[1]))
    v = centres + e
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load(
    conn: psycopg.Connection, n: int, dim: int, centres: np.ndarray, noise: float, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS ann_bench")
        cur.execute(f"CREATE TABLE ann_bench (id bigserial PRIMARY KEY, embedding vector({dim}))")
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        with cur.copy("COPY ann_bench (embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["vector"])
            done = 0
            while done < n:
                m = min(CHUNK, n - done)
                # кадр i принадлежит машине i mod len(centres)
                owners = centres[(np.arange(done, done + m)) % len(centres)]
                for row in views(rng, owners, noise):
                    copy.write_row([row])
                done += m
                if done % 200_000 == 0 or done >= n:
                    print(f"  загружено {done:,} за {time.perf_counter() - t0:.0f} с")
    conn.commit()
    return time.perf_counter() - t0


def topk(conn: psycopg.Connection, q: np.ndarray, k: int) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM ann_bench ORDER BY embedding <=> %s LIMIT %s", (q, k), binary=True
        )
        ids = [r[0] for r in cur.fetchall()]
    return ids, (time.perf_counter() - t0) * 1000


def percentiles(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=None, help="по умолчанию dim из models/model.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--exact-queries", type=int, default=20, help="перебор для recall и сравнения")
    p.add_argument("--ef-search", default="40,100", help="значения hnsw.ef_search через запятую")
    p.add_argument("--views", type=int, default=6, help="кадров на машину (как в датасете)")
    p.add_argument("--noise", type=float, default=0.5, help="норма шума: cos к центру ≈ 1/sqrt(1+s²)")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", action="store_true", help="не удалять таблицу после замера")
    p.add_argument("--out", type=Path, default=ROOT / "bench" / "results")
    args = p.parse_args()
    dim = args.dim or json.loads((ROOT / "models" / "model.json").read_text())["dim"]
    url = os.environ.get("DATABASE_URL", "postgres://reid:reid@localhost:5432/reid")

    conn = psycopg.connect(url, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)

    n_vehicles = max(args.n // args.views, 1)
    print(f"N={args.n:,} dim={dim}, машин {n_vehicles:,} × {args.views} → {url.rsplit('@', 1)[-1]}")
    rng = np.random.default_rng(args.seed + 1)
    centres = unit_vectors(rng, n_vehicles, dim)
    load_s = load(conn, args.n, dim, centres, args.noise, args.seed)
    # запросы — новые кадры случайных машин из галереи
    queries = views(rng, centres[rng.choice(n_vehicles, args.queries)], args.noise)

    # Эталон перебором: до индекса планировщик может выбрать только seq scan
    print("перебор без индекса …")
    truth, exact_ms = [], []
    for q in queries[: args.exact_queries]:
        ids, ms = topk(conn, q, args.k)
        truth.append(set(ids))
        exact_ms.append(ms)

    print("строим HNSW …")
    with conn.cursor() as cur:
        # Граф 10^6 × 512-d ≈ 2.3 ГБ; при нехватке pgvector строит в несколько проходов.
        # Параллельная сборка требует /dev/shm не меньше этого значения (shm_size в compose).
        cur.execute("SET maintenance_work_mem = '2GB'")
        cur.execute("SET max_parallel_maintenance_workers = 7")
        t0 = time.perf_counter()
        cur.execute(
            "CREATE INDEX ann_bench_hnsw ON ann_bench USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
        )
        build_s = time.perf_counter() - t0
        # total: векторы 2 КБ лежат в TOAST, pg_relation_size их не видит
        cur.execute(
            "SELECT pg_total_relation_size('ann_bench') - pg_relation_size('ann_bench_hnsw'), "
            "pg_relation_size('ann_bench_hnsw')"
        )
        table_bytes, index_bytes = cur.fetchone()
    conn.commit()
    print(f"  индекс за {build_s:.0f} с, таблица {table_bytes / 2**30:.2f} ГБ, индекс {index_bytes / 2**30:.2f} ГБ")

    for q in queries[:20]:  # прогрев кэша страниц индекса, иначе первый ef_search медленнее
        topk(conn, q, args.k)
    runs = []
    for ef in [int(x) for x in args.ef_search.split(",")]:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {ef}")
        ms, hits = [], []
        for i, q in enumerate(queries):
            ids, t = topk(conn, q, args.k)
            ms.append(t)
            if i < len(truth):
                hits.append(len(set(ids) & truth[i]) / args.k)
        runs.append({"ef_search": ef, **percentiles(ms), "recall_at_k": float(np.mean(hits))})
        print(f"  ef_search={ef}: {runs[-1]}")

    result = {
        "n": args.n,
        "dim": dim,
        "k": args.k,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "views_per_vehicle": args.views,
        "noise": args.noise,
        "load_s": load_s,
        "index_build_s": build_s,
        "table_gb": table_bytes / 2**30,
        "index_gb": index_bytes / 2**30,
        "exact": percentiles(exact_ms),
        "hnsw": runs,
        "pgvector": conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"ann_n{args.n}_d{dim}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"→ {out}")
    print("\nСтрока для 05-scaling.md:")
    best = runs[-1]
    print(
        f"| {args.n:,} × {dim}-d | загрузка {load_s:.0f} с | HNSW m={args.m}, ef_c={args.ef_construction}: "
        f"{build_s:.0f} с, {index_bytes / 2**30:.1f} ГБ | перебор p50 {result['exact']['p50_ms']:.0f} мс | "
        f"HNSW ef={best['ef_search']}: p50 {best['p50_ms']:.1f} мс, p95 {best['p95_ms']:.1f} мс, "
        f"recall@{args.k} {best['recall_at_k']:.2f} |"
    )

    if not args.keep:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE ann_bench")
        conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
