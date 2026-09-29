"""CPU checks of the restored training path: manifests, recipe configs, 2+2 sampler, VeRi-Wild list."""

import hashlib
import math
from collections import Counter
from pathlib import Path

import pytest

from train import fit_v8, stage1_veriwild, stage2_fit

TRAIN = Path(__file__).resolve().parents[1] / "train"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_restored_sources_are_the_ones_the_shipped_run_recorded():
    # Hashes recorded by run vitl_fit_2plus2_v1_e8 and its export (ledger / export.json in 058c348^).
    assert sha256(TRAIN / "reid_model.py") == "c715786b2acec7b7c8a7581df425d132d0dec2f4ee7e952072adf80ed9d0ee49"
    assert sha256(TRAIN / "checkpoint_inference.py") == "f8f7d7e431247baa9b1408e537f40055894de48c0d545031ca9d9d2eb8f386a2"
    assert sha256(TRAIN / "residual.py") == "4587562390e53fad81680b19640f37f080e8ed3b5922f01a9f11a4ec0dc3f1f8"


def test_fit_v8_is_the_documented_derivation_of_organizer_rows():
    organizer = fit_v8.ORGANIZER_TRAIN_CSV if fit_v8.ORGANIZER_TRAIN_CSV.is_file() else None
    report = fit_v8.check(organizer)
    assert (report["fit_rows"], report["fit_v8_rows"], report["fit_v8_identities"]) == (5746, 5722, 928)


def test_fit_v8_identities_are_disjoint_from_search():
    stage2_fit.check_fit_input(stage2_fit.FIT_V8_CSV, stage2_fit.SEARCH_DIR)


def test_stage_configs_reproduce_recorded_update_counts(tmp_path):
    stage1 = stage1_veriwild.parse_args(["--config", str(TRAIN / "configs/stage1_veriwild.json"),
                                         "--run-dir", str(tmp_path / "s1")])
    stage2 = stage2_fit.parse_args(["--config", str(TRAIN / "configs/stage2_fit_v8_2plus2.json"),
                                    "--run-dir", str(tmp_path / "s2")])
    for args in (stage1, stage2):
        assert (args.model, args.image_size, args.resize_mode, args.global_pool) == (
            "vit_large_patch16_dinov3.lvd1689m", 256, "stretch", "token")
        assert (args.batch_size, args.samples_per_id, args.seed, args.amp) == (16, 4, 20260920, "bf16")
        assert (args.lr, args.head_lr, args.weight_decay, args.warmup_steps) == (3e-5, 3e-4, 0.05, 50)
        assert (args.margin, args.label_smoothing, args.triplet_weight) == (0.3, 0.1, 1.0)
    assert stage1.epochs * math.ceil(stage1_veriwild.EXPECTED["images"] / stage1.batch_size) == 34726
    assert stage2.epochs * math.ceil(5722 / stage2.batch_size) == 2864
    assert (stage2.identity_sampler, stage2.workers, stage1.workers) == ("fit_2plus2_v1", 2, 0)
    assert stage2_fit.parse_args(["--run-dir", "x", "--config", str(TRAIN / "configs/stage2_fit_v8_2plus2.json"),
                                  "--epochs", "1"]).epochs == 1


def test_veriwild_train_list_parser():
    rows = stage1_veriwild.parse_train_list(
        "path label camera\nimages/37967/000001.jpg 1 102\n37967/000002 1 5\n23488/000003.jpg,0,7\n")
    assert [r["path"] for r in rows] == ["37967/000001.jpg", "37967/000002.jpg", "23488/000003.jpg"]
    assert [r["vehicle_id"] for r in rows] == ["veriwild:37967", "veriwild:37967", "veriwild:23488"]
    assert [r["label"] for r in rows] == [1, 1, 0]
    assert stage1_veriwild.describe(rows) == {"images": 3, "identities": 2, "cameras": 3}
    with pytest.raises(ValueError, match="contiguous"):
        stage1_veriwild.parse_train_list("1/a.jpg 0 1\n2/b.jpg 2 1\n")
    with pytest.raises(ValueError, match="several identity directories"):
        stage1_veriwild.parse_train_list("1/a.jpg 0 1\n2/b.jpg 0 1\n")
    with pytest.raises(ValueError, match="unsafe"):
        stage1_veriwild.parse_train_list("../x/a.jpg 0 1\n")


def test_2plus2_sampler_rebalances_only_eligible_three_plus_one_blocks():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from train.reid_model import FitTwoPlusTwoBatches, IdentityBatches, self_check

    self_check()
    rows = []
    for identity in range(12):
        # Even identities: two cameras with 4 + 2 images (eligible); odd: one camera only.
        cameras = {"a": 4, "b": 2} if identity % 2 == 0 else {"a": 5}
        rows += [{"vehicle_id": str(identity), "camera_id": f"{identity}{camera}"}
                 for camera, count in cameras.items() for _ in range(count)]
    parent = list(IdentityBatches(rows, 16, 4, 20260920, 3))
    child = list(FitTwoPlusTwoBatches(rows, 16, 4, 20260920, 3))
    assert child == list(FitTwoPlusTwoBatches(rows, 16, 4, 20260920, 3))
    changed = 0
    for before, after in zip(parent, child, strict=True):
        assert len(after) == 16
        for offset in range(0, 16, 4):
            old, new = before[offset:offset + 4], after[offset:offset + 4]
            identity = int(rows[old[0]]["vehicle_id"])
            old_counts = sorted(Counter(rows[i]["camera_id"] for i in old).values())
            if identity % 2 == 0 and old_counts == [1, 3]:
                assert sorted(Counter(rows[i]["camera_id"] for i in new).values()) == [2, 2]
                assert len(set(new)) == 4 and len(set(old) & set(new)) == 3
                changed += 1
            else:
                assert new == old
    assert changed > 0


def test_veriwild_dry_run_reports_coverage(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from PIL import Image

    lines = []
    for label in range(4):
        for image in range(4):
            path = tmp_path / "images" / f"{label + 100}" / f"{image}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)
            lines.append(f"images/{label + 100}/{image}.jpg {label} {image % 2}")
    (tmp_path / "train_test_split").mkdir()
    (tmp_path / "train_test_split/train_list_start0.txt").write_text("\n".join(lines) + "\n")
    stage1_veriwild.main(["--run-dir", str(tmp_path / "run"), "--veriwild-root", str(tmp_path),
                          "--dry-run", "--allow-count-mismatch", "--batch-size", "8", "--epochs", "2"])
    output = capsys.readouterr().out
    assert "'images': 16" in output and "after_epoch_2" in output
    assert not (tmp_path / "run").exists()
