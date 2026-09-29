"""Stage 1: vehicle-domain training of DINOv3 ViT-L/16 on the official VeRi-Wild TRAIN partition.

RECONSTRUCTION. The historical external-stage runner (artifacts/vitl_wild_pretrain_v1 and
artifacts/vitl_wild_e2_v1/run_wild_pretrain.py) was never committed to git. This entry point
re-implements its documented contract with the restored model/sampler/loss code: public DINOv3
initialization, a fresh 30,671-class BN/classifier head, uniform camera-diverse P4xK4 batches,
2 nominal epochs = 2 x 17,363 = 34,726 updates, workers 0, the shared recipe, and export of the
terminal backbone only (backbone.safetensors) as the stage-2 initialization.
"""

import os
import re
from pathlib import Path

from train import trainer

EXPECTED = {"images": 277797, "identities": 30671, "cameras": 173}
FOUNDATION_SHA256 = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"


def parse_train_list(text):
    """Rows of the official train list: image path, contiguous label 0..N-1, camera id.

    Accepts whitespace, comma or semicolon separators and an optional header line. Paths may be
    `images/<identity>/<image>.jpg`, `<identity>/<image>.jpg` or without the `.jpg` suffix.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        fields = [field for field in re.split(r"[\s,;]+", line.strip()) if field]
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"Line {number}: expected 'path label camera', got {line!r}")
        path, label, camera = fields
        if not label.isdigit():
            if rows or number != 1:
                raise ValueError(f"Line {number}: non-numeric label {label!r}")
            continue  # header
        path = path.removeprefix("images/")
        path = path if path.endswith(".jpg") else path + ".jpg"
        if path.startswith("/") or ".." in Path(path).parts or len(Path(path).parts) != 2:
            raise ValueError(f"Line {number}: unsafe or unexpected image path {path!r}")
        namespace = "veriwild:" + Path(path).parts[0]
        # The historical sampler sorts identity names, not numeric classifier labels.
        rows.append({"path": path, "vehicle_id": namespace, "label": int(label),
                     "camera_id": camera, "identity_namespace": namespace})
    labels = sorted({row["label"] for row in rows})
    if labels != list(range(len(labels))):
        raise ValueError("Official training labels must be contiguous from 0")
    namespaces = {}
    for row in rows:
        if namespaces.setdefault(row["label"], row["vehicle_id"]) != row["vehicle_id"]:
            raise ValueError(f"Label {row['label']} maps to several identity directories")
    if len(set(namespaces.values())) != len(namespaces):
        raise ValueError("Label to identity-directory mapping is not one-to-one")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image paths in the training list")
    return rows


def describe(rows):
    return {"images": len(rows), "identities": len({r["vehicle_id"] for r in rows}),
            "cameras": len({r["camera_id"] for r in rows})}


class WholeImages:
    """VeRi-Wild images are already vehicle crops: use the full RGB image."""
    def __init__(self, rows, images_dir, transform):
        self.rows, self.images_dir, self.transform = rows, Path(images_dir), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(self.images_dir / row["path"]) as image:
            return self.transform(image.convert("RGB")), row["label"]


def coverage(rows, sampler_class, args, epochs):
    """Cumulative unique images/identities presented by the sampler after each nominal epoch.

    Historical records: 144,710 images / 27,528 identities after epoch 1 and 189,014 / 30,383
    after epoch 2. Equal values show that list order, labels and sampler match the original run.
    """
    seen, report = set(), {}
    for epoch in range(epochs):
        for batch in sampler_class(rows, args.batch_size, args.samples_per_id, args.seed, epoch):
            seen.update(batch)
        report[f"after_epoch_{epoch + 1}"] = {"unique_images": len(seen),
                                             "unique_identities": len({rows[i]["vehicle_id"] for i in seen})}
    return report


def parse_args(argv=None):
    parser = trainer.new_parser(__doc__)
    trainer.add_recipe_arguments(parser)
    parser.add_argument("--veriwild-root", type=Path, default=os.environ.get("VERIWILD_ROOT"),
                        help="VeRi-Wild root with images/ and train_test_split/ (env VERIWILD_ROOT)")
    parser.add_argument("--train-list", type=Path,
                        help="Official train list; default <root>/train_test_split/train_list_start0.txt")
    parser.add_argument("--images-dir", type=Path, help="Default <root>/images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate the list/images and print sampler coverage; no GPU, no training")
    parser.add_argument("--allow-count-mismatch", action="store_true",
                        help="Do not require the official 277,797/30,671/173 counts (smoke runs only)")
    return trainer.parse_with_config(parser, argv)


def main(argv=None):
    args = parse_args(argv)
    if args.veriwild_root is None:
        raise SystemExit("Set --veriwild-root or VERIWILD_ROOT")
    train_list = args.train_list or args.veriwild_root / "train_test_split/train_list_start0.txt"
    images_dir = args.images_dir or args.veriwild_root / "images"
    rows = parse_train_list(train_list.read_text(encoding="utf-8-sig"))
    counts = describe(rows)
    if counts != EXPECTED and not args.allow_count_mismatch:
        raise ValueError(f"Not the official VeRi-Wild TRAIN partition: {counts} != {EXPECTED}")
    missing = [row["path"] for row in rows if not (images_dir / row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed images are missing, e.g. {missing[:3]}")
    if args.dry_run:
        from train.reid_model import IdentityBatches
        print({**counts, "coverage": coverage(rows, IdentityBatches, args, args.epochs)})
        return
    if args.weights is None and args.resume is None:
        raise SystemExit("--weights (DINOv3 foundation .safetensors) or --resume is required")
    trainer.require_cuda(args.amp)
    from safetensors.torch import save_file

    from train.reid_model import IdentityBatches, build_model, make_transforms, seed_everything
    seed_everything(args.seed)
    labels = {row["vehicle_id"]: row["label"] for row in rows}
    config = trainer.model_config(args, labels)
    config["input_hashes"] = {str(path.resolve()): trainer.sha256(path)
                              for path in [train_list, *([args.weights] if args.weights else [])]}
    config["data"] = counts
    checkpoint = trainer.load_resume(args.resume)
    if checkpoint is not None:
        config["input_hashes"] = checkpoint["config"]["input_hashes"]
    elif config["input_hashes"][str(args.weights.resolve())] != FOUNDATION_SHA256:
        print(f"WARNING: {args.weights} differs from the pinned DINOv3 revision ({FOUNDATION_SHA256})")
    sources = [trainer.HERE / "reid_model.py", trainer.HERE / "trainer.py", Path(__file__).resolve()]
    run = trainer.prepare_run(args, config, checkpoint, sources)
    model = build_model(config, weights=None if checkpoint else args.weights, checkpoint=checkpoint).cuda()
    train_transform, _, preprocessing = make_transforms(model, args.image_size, args.resize_mode)
    trainer.write_json(run / "preprocessing.json", preprocessing)
    last = trainer.train(run, model, WholeImages(rows, images_dir, train_transform), rows,
                         IdentityBatches, args, config, checkpoint)
    if last["epoch"] < args.epochs:
        return
    # Only the backbone initializes stage 2; the 30,671-class head and BN neck are discarded.
    backbone = {key: value.detach().cpu().contiguous() for key, value in model.backbone.state_dict().items()}
    save_file(backbone, str(run / "backbone.safetensors"))
    report = {"backbone_sha256": trainer.sha256(run / "backbone.safetensors"), "checkpoint": last,
              "coverage": coverage(rows, IdentityBatches, args, args.epochs), "data": counts}
    trainer.write_json(run / "backbone.json", report)
    trainer.log_event(run, event="backbone_exported", **report)


if __name__ == "__main__":
    main()
