"""Create submission.csv, embeddings.npy and candidates.csv from images and crop CSVs.

Run from ml/ (or deploy/submit.Dockerfile):
  uv run python -m reid.submit --images ../dataset/images --query ../dataset/test_query.csv \
      --gallery ../dataset/test_gallery.csv --out ../submission

Each query is independent. The model manifest selects static-gallery DBA and its
matching rejection threshold. Original embeddings remain unmodified in the artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from extractor import Extractor
from reid.artifacts import submission_retrieval, write_artifacts


def embed_csv(ex: Extractor, csv: Path, images: Path, batch: int) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv, dtype={"image_id": str})
    items = [
        (images / f"{r.image_id}.jpg", (int(r.x), int(r.y), int(r.w), int(r.h)))
        for r in df.itertuples(index=False)
    ]
    out = [ex.extract_batch(items[i : i + batch]) for i in range(0, len(items), batch)]
    return df.image_id.tolist(), np.concatenate(out) if out else np.zeros((0, ex.dim), np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--gallery", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("submission"))
    p.add_argument("--manifest", type=Path, default=None, help="models/model.json")
    p.add_argument("--threshold", type=float, default=None, help="переопределить порог манифеста")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    ex = Extractor(args.manifest) if args.manifest else Extractor()
    ex.warmup()
    use_dba, threshold = submission_retrieval(ex.m, args.threshold)
    print(f"модель {ex.m['name']} {ex.m['version']}, {ex.provider}, порог {threshold}")

    t0 = time.perf_counter()
    q_ids, q_emb = embed_csv(ex, args.query, args.images, args.batch_size)
    g_ids, g_emb = embed_csv(ex, args.gallery, args.images, args.batch_size)
    secs = time.perf_counter() - t0
    n = len(q_ids) + len(g_ids)
    print(
        f"{n} кропов за {secs:.1f} с: {n / secs:.1f} кроп/с (батч {args.batch_size}, полный цикл)"
    )

    paths = write_artifacts(args.out, q_ids, g_ids, q_emb, g_emb, threshold, use_dba=use_dba)
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "model": ex.m["name"],
                "version": ex.m["version"],
                "provider": ex.provider,
                "threshold": threshold,
                "retrieval": "gallery_dba_k4_alpha2" if use_dba else "cosine",
                "n_query": len(q_ids),
                "n_gallery": len(g_ids),
                "batch_size": args.batch_size,
                "seconds": round(secs, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
