"""Инварианты валидационного split-а по протоколу docs/03-dataset.md §6."""

import numpy as np
import pandas as pd

from reid.split import make_split


def synthetic_train(n_ids: int = 20, seed: int = 0) -> pd.DataFrame:
    """Как train.csv: у каждой машины 2–4 камеры, на (машина, камера) серия из 1–3 кадров."""
    rng = np.random.default_rng(seed)
    rows = []
    for vid in range(n_ids):
        for cam in rng.choice(30, size=rng.integers(2, 5), replace=False):
            for k in range(rng.integers(1, 4)):
                rows.append(
                    {
                        "image_id": f"{vid}_{cam}_{k}",
                        "x": 0,
                        "y": 0,
                        "w": 10,
                        "h": 10,
                        "vehicle_id": vid,
                        "camera_id": int(cam),
                    }
                )
    return pd.DataFrame(rows)


def test_holds_out_requested_ids_and_keeps_the_rest_for_training():
    df = synthetic_train()
    out = make_split(df, n_ids=8, no_gallery_frac=0.25, seed=1)
    val = out[out.split != "train"]
    assert len(out) == len(df)
    assert val.vehicle_id.nunique() == 8
    assert set(out[out.split == "train"].vehicle_id).isdisjoint(set(val.vehicle_id))


def test_gallery_is_one_camera_and_queries_come_from_other_cameras():
    out = make_split(synthetic_train(), n_ids=8, no_gallery_frac=0.25, seed=1)
    for _, rows in out[out.split != "train"].groupby("vehicle_id"):
        gallery = rows[rows.split == "gallery"]
        query = rows[rows.split == "query"]
        assert len(query) > 0
        if len(gallery):
            assert gallery.camera_id.nunique() == 1
            assert not set(gallery.camera_id) & set(query.camera_id)


def test_requested_fraction_of_ids_has_no_gallery_for_tnr():
    out = make_split(synthetic_train(), n_ids=8, no_gallery_frac=0.25, seed=1)
    val = out[out.split != "train"]
    with_gallery = val[val.split == "gallery"].vehicle_id.nunique()
    assert val.vehicle_id.nunique() - with_gallery == 2


def test_split_is_deterministic_for_a_seed():
    df = synthetic_train()
    a = make_split(df, n_ids=8, no_gallery_frac=0.25, seed=3)
    b = make_split(df, n_ids=8, no_gallery_frac=0.25, seed=3)
    pd.testing.assert_frame_equal(a, b)
