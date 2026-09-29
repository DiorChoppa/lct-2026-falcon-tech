"""Валидационный split по протоколу docs/03-dataset.md §6.

Из train.csv откладываем n_ids машин. У каждой одна камера (вся серия) идёт в галерею,
остальные камеры — в запросы, поэтому пары query–gallery всегда кросс-камерные, как в тесте.
Доля no_gallery_frac отложенных машин в галерею не попадает совсем: их кадры — запросы без
правильного ответа, на них меряется TNR и подбирается порог. Остальные машины — "train".

Запуск из ml/: uv run python -m reid.split [--seed 0] → val_split.csv (все строки train.csv
плюс столбец split ∈ {train, query, gallery}).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def make_split(
    df: pd.DataFrame, n_ids: int = 300, no_gallery_frac: float = 0.25, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    val_ids = rng.choice(np.sort(df["vehicle_id"].unique()), size=n_ids, replace=False)
    no_gallery = set(val_ids[: round(n_ids * no_gallery_frac)].tolist())

    split = pd.Series("train", index=df.index)
    for vid in val_ids:
        rows = df.index[df["vehicle_id"] == vid]
        if vid in no_gallery:
            split[rows] = "query"
            continue
        cams = np.sort(df.loc[rows, "camera_id"].unique())
        if cams.size < 2:
            raise ValueError(f"у машины {vid} одна камера, кросс-камерный запрос невозможен")
        is_gallery = (df.loc[rows, "camera_id"] == rng.choice(cams)).to_numpy()
        split[rows] = np.where(is_gallery, "gallery", "query")

    out = df.copy()
    out["split"] = split.to_numpy()
    return out


def describe(out: pd.DataFrame) -> str:
    val = out[out["split"] != "train"]
    query, gallery = val[val["split"] == "query"], val[val["split"] == "gallery"]
    with_pair = query["vehicle_id"].isin(gallery["vehicle_id"])
    return (
        f"train: {out[out['split'] == 'train']['vehicle_id'].nunique()} машин, "
        f"{(out['split'] == 'train').sum()} кадров\n"
        f"val:   {val['vehicle_id'].nunique()} машин; gallery {len(gallery)} кадров "
        f"({gallery['vehicle_id'].nunique()} машин); query {len(query)} кадров, "
        f"из них с парой {int(with_pair.sum())}, без пары {int((~with_pair).sum())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=Path, default=ROOT / "dataset" / "train.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "val_split.csv")
    parser.add_argument("--n-ids", type=int, default=300)
    parser.add_argument("--no-gallery-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = make_split(pd.read_csv(args.train), args.n_ids, args.no_gallery_frac, args.seed)
    out.to_csv(args.out, index=False)
    print(describe(out))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
