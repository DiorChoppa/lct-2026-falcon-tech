"""Замер скорости по протоколу жюри (07-organizers-qa.md §1, вопрос 31).

latency_b1 — медиана полного цикла extract() при batch=1: 300 прогонов после 50 прогревочных
(session.run возвращает выход на хосте, т. е. CUDA уже синхронизирована).
throughput — устойчивый FPS extract_batch при батчах 1/8/16/32, не короче 10 с на размер;
перед замером каждого размера — прогревочные батчи (новая форма входа инициализирует CUDA EP).
Баллы: latency ≤ 40 мс и FPS ≥ 100 — полный балл; 80 мс и 50 FPS — ноль.
--embeddings сохраняет выходы замера latency (строка i — кадр i из CSV): два прогона
с побитным сравнением файлов — проверка детерминизма (docs/speed.md).

  uv run python -m reid.perf --images ../dataset/images --csv ../dataset/test_query.csv
  (на GPU в образе жюри — just perf-gpu, docs/speed.md)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from extractor import Extractor


def latency_score(ms: float) -> float:
    return float(np.clip((80 - ms) / 40, 0, 1))


def throughput_score(fps: float) -> float:
    return float(np.clip((fps - 50) / 50, 0, 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True, help="CSV с image_id,x,y,w,h")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--batches", default="1,8,16,32", help="пусто — без замера throughput")
    p.add_argument("--batch-warmup", type=int, default=50, help="прогревочных батчей на размер")
    p.add_argument("--embeddings", type=Path, default=None, help=".npy с выходами замера latency")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    # csv, а не pandas: скрипт запускается и в образе жюри (deploy/submit.Dockerfile)
    with open(args.csv, newline="") as f:
        items = [
            (args.images / f"{r['image_id']}.jpg", tuple(int(r[k]) for k in "xywh"))
            for r in csv.DictReader(f)
        ]
    t0 = time.perf_counter()
    ex = Extractor(args.manifest) if args.manifest else Extractor()
    if ex.provider != "CUDAExecutionProvider":
        raise RuntimeError("Jury speed measurements require CUDAExecutionProvider")
    load_s = time.perf_counter() - t0
    print(f"модель {ex.m['name']} {ex.m['version']}, {ex.provider}, загрузка {load_s:.1f} с")

    for i in range(args.warmup):
        ex.extract(*items[i % len(items)])
    times, outs = [], []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = ex.extract(*items[i % len(items)])
        times.append(time.perf_counter() - t0)
        outs.append(out)
    if args.embeddings:
        np.save(args.embeddings, np.stack(outs))
    lat_ms = float(np.median(times) * 1000)
    print(f"latency_b1: медиана {lat_ms:.1f} мс, p95 {np.percentile(times, 95) * 1000:.1f} мс")

    fps = {}
    k = 0
    for b in (int(v) for v in args.batches.split(",") if v):
        for _ in range(args.batch_warmup):
            ex.extract_batch([items[(k + j) % len(items)] for j in range(b)])
            k += b
        done, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            batch = [items[(k + j) % len(items)] for j in range(b)]
            ex.extract_batch(batch)
            done, k = done + b, k + b
        fps[b] = done / (time.perf_counter() - t0)
        print(f"batch {b:>2}: {fps[b]:.1f} FPS")

    best = max(fps.values(), default=0.0)
    report = {
        "protocol": {"latency_warmups": args.warmup, "latency_runs": args.runs,
                     "throughput_warmups_per_shape": args.batch_warmup,
                     "throughput_minimum_seconds_per_shape": args.seconds},
        "model": ex.m["name"],
        "version": ex.m["version"],
        "provider": ex.provider,
        "load_s": round(load_s, 2),
        "latency_b1_ms": round(lat_ms, 2),
        "latency_p95_ms": float(np.percentile(times, 95) * 1000),
        "latency_samples_ms": [float(t * 1000) for t in times],
        "fps": {str(k): round(v, 1) for k, v in fps.items()},
        "best_fps": round(best, 1),
        "latency_score": round(latency_score(lat_ms), 3),
        "throughput_score": round(throughput_score(best), 3),
    }
    print(
        f"баллы: latency {report['latency_score']:.2f}, throughput {report['throughput_score']:.2f}"
        f" (из 1.0 каждый, по 10% оценки)"
    )
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
