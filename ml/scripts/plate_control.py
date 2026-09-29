"""Контрольный прогон «номер залит сплошным цветом» (ответ организаторов 48).

Финалистов прогоняют на версии теста, где зона номера залита; заметное падение метрики —
признак опоры на остаточный сигнал номера. Координат номеров у нас нет, поэтому зона
находится по пикселизации (`reid.plate_mask`) на каждом кропе query и gallery.

Три условия на одном эпизоде, всё остальное (модель, препроцессинг, DBA, порог) неизменно:
  original — исходные кропы;
  plate    — найденные зоны номера залиты чёрным;
  control  — прямоугольники того же размера залиты в случайном месте кропа, не на номере.
plate ≈ control ≈ original — модель не опирается на номер; plate ≪ control — опирается.

  uv run python scripts/plate_control.py --runs <dir с <episode>/embeddings.npy> \
      --episodes search audit --work /tmp/plate --json validation/plate_control.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))

import eval_holdout as eh

from extractor import Extractor
from reid.artifacts import l2
from reid.plate_mask import control_boxes, plate_boxes
from reid.preprocess import crop_bbox
from reid.submit import embed_csv

CONDITIONS = ("plate", "control")


def prepare(episode: str, images: Path, work: Path) -> dict:
    """Кропы трёх условий как PNG (имя .jpg — так их читает embed_csv) и CSV с bbox на весь кроп."""
    root = ML / "validation" / episode
    record = {}
    for part in ("query", "gallery"):
        df = pd.read_csv(root / f"{part}.csv", dtype={"image_id": str})
        rows = {c: [] for c in ("original", *CONDITIONS)}
        for r in df.itertuples(index=False):
            with Image.open(images / f"{r.image_id}.jpg") as frame:
                crop = crop_bbox(frame.convert("RGB"), r.x, r.y, r.w, r.h)
            boxes = plate_boxes(np.asarray(crop))
            seed = int(hashlib.sha256(r.image_id.encode()).hexdigest()[:8], 16)
            ctrl = control_boxes(boxes, crop.size, seed)
            record[r.image_id] = {"plate": boxes, "control": ctrl, "size": crop.size}
            for cond, bx in (("original", []), ("plate", boxes), ("control", ctrl)):
                out = work / episode / cond
                out.mkdir(parents=True, exist_ok=True)
                img = crop.copy()
                draw = ImageDraw.Draw(img)
                for b in bx:
                    draw.rectangle((b[0], b[1], b[2] - 1, b[3] - 1), fill=(0, 0, 0))
                img.save(out / f"{r.image_id}.jpg", format="PNG")
                rows[cond].append(
                    {"image_id": r.image_id, "x": 0, "y": 0, "w": crop.width, "h": crop.height}
                )
        for cond, rr in rows.items():
            pd.DataFrame(rr).to_csv(work / episode / f"{cond}_{part}.csv", index=False)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, required=True, help="исходные эмбеддинги эпизодов")
    p.add_argument("--images", type=Path, default=ML.parent / "dataset" / "images")
    p.add_argument("--episodes", nargs="+", default=["search", "audit"])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    ex = Extractor()
    manifest = ex.m
    report = {"model": manifest["name"], "fill": "solid black (0,0,0)", "episodes": {}}
    for episode in args.episodes:
        record = prepare(episode, args.images, args.work)
        q_ids, g_ids, q0, g0, query, gallery = eh.load(args.runs, episode)
        # PNG-кроп без заливки обязан давать тот же вектор, что исходный JPEG-кадр
        _, check = embed_csv(
            ex,
            args.work / episode / "original_query.csv",
            args.work / episode / "original",
            args.batch_size,
        )
        parity = float(np.min(np.sum(l2(check) * q0, axis=1)))
        emb = {"original": (q0, g0)}
        for cond in CONDITIONS:
            d = args.work / episode / cond
            _, qe = embed_csv(ex, args.work / episode / f"{cond}_query.csv", d, args.batch_size)
            _, ge = embed_csv(ex, args.work / episode / f"{cond}_gallery.csv", d, args.batch_size)
            emb[cond] = (l2(qe), l2(ge))

        masked = [i for i, v in record.items() if v["plate"]]
        area = [
            sum((b[2] - b[0]) * (b[3] - b[1]) for b in v["plate"]) / (v["size"][0] * v["size"][1])
            for v in record.values()
            if v["plate"]
        ]
        res = {
            "crops": len(record),
            "crops_with_plate_zone": len(masked),
            "query_crops_with_plate_zone": sum(bool(record[q]["plate"]) for q in q_ids),
            "masked_area_fraction_mean": float(np.mean(area)) if area else 0.0,
            "unmasked_png_min_cosine_to_original": parity,
            "conditions": {},
        }
        for method, use_dba, thr in (
            ("dba", True, float(manifest["retrieval"]["threshold"])),
            ("cosine", False, float(manifest["threshold"])),
        ):
            for cond, (qe, ge) in emb.items():
                rk, top1 = eh.rank(qe, ge, g_ids, use_dba)
                res["conditions"][f"{method}/{cond}"] = eh.evaluate(
                    q_ids, rk, top1, thr, query, gallery
                )
        # насколько сдвинулся вектор запроса от заливки: 1 − cos, по запросам с найденным номером
        has = np.array([bool(record[q]["plate"]) for q in q_ids])
        for cond in CONDITIONS:
            shift = 1 - np.sum(emb[cond][0] * q0, axis=1)
            res[f"query_shift_{cond}"] = float(shift[has].mean()) if has.any() else 0.0
        report["episodes"][episode] = res

        fmt = "{:<8} {:<9} {:>7} {:>7} {:>7} {:>7}  {}"
        print(
            f"\n{episode}: зона номера найдена на {len(masked)}/{len(record)} кропов, "
            f"средняя доля площади {res['masked_area_fraction_mean']:.3f}, паритет PNG {parity:.6f}"
        )
        print(fmt.format("метод", "условие", "mAP@10", "R1", "F1", "Q", "TP/FP/FN/TN"))
        for key, r in res["conditions"].items():
            method, cond = key.split("/")
            print(
                fmt.format(
                    method,
                    cond,
                    *(f"{r[k]:.4f}" for k in ("mAP@10", "Rank-1", "F1", "Q")),
                    f"{r['TP']}/{r['FP']}/{r['FN']}/{r['TN']}",
                )
            )
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
