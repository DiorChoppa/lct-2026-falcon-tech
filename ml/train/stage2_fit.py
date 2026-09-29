"""Stage 2: contest FIT fine-tuning of the VeRi-Wild backbone on FIT-v8 with the 2+2 sampler.

Restored from the historical scripts/run_experiment.py (sha256 58518bf5..., the source hash
recorded by the shipped run vitl_fit_2plus2_v1_e8). Only the shipped treatment is kept:
fresh 928-class BN/classifier head, `--identity-sampler fit_2plus2_v1`, 8 epochs = 2,864
updates. SEARCH evaluation, campaign deadlines and other research treatments were removed;
they ran after the terminal checkpoint was written and did not affect its weights.
"""

import csv
from pathlib import Path

from train import trainer

HERE = Path(__file__).resolve().parent
FIT_V8_CSV = HERE / "data/fit_v8/train.csv"
FIT_V8_SHA256 = "fe1394a9fbc7ba5e21eeed802252d373348940ebece6d0ae5a6c4727215ff962"
SEARCH_DIR = HERE.parent / "validation/search"


def read_identities(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return {row["vehicle_id"] for row in csv.DictReader(stream)}


def check_fit_input(train_csv, search_dir):
    """Only the frozen FIT-v8 manifest; its identities must be disjoint from SEARCH."""
    if trainer.sha256(train_csv) != FIT_V8_SHA256:
        raise ValueError(f"{train_csv} is not the frozen FIT-v8 manifest ({FIT_V8_SHA256})")
    evaluated = set()
    for name in ("query.csv", "gallery.csv"):
        if (Path(search_dir) / name).is_file():
            evaluated |= read_identities(Path(search_dir) / name)
    if read_identities(train_csv) & evaluated:
        raise ValueError("FIT and SEARCH identities overlap")


def parse_args(argv=None):
    parser = trainer.new_parser(__doc__)
    trainer.add_recipe_arguments(parser)
    parser.add_argument("--train-csv", type=Path, default=FIT_V8_CSV)
    parser.add_argument("--images-dir", type=Path, default=trainer.REPO / "dataset/images",
                        help="Organizer full-frame JPEGs (<image_id>.jpg)")
    parser.add_argument("--search-dir", type=Path, default=SEARCH_DIR,
                        help="SEARCH query/gallery CSVs used only for the identity-overlap guard")
    parser.add_argument("--identity-sampler", choices=["uniform", "fit_2plus2_v1"], default="fit_2plus2_v1")
    return trainer.parse_with_config(parser, argv)


def main(argv=None):
    args = parse_args(argv)
    if args.weights is None and args.resume is None:
        raise SystemExit("--weights (stage-1 backbone.safetensors) or --resume is required")
    check_fit_input(args.train_csv, args.search_dir)
    trainer.require_cuda(args.amp)
    from train.reid_model import (
        FitTwoPlusTwoBatches,
        IdentityBatches,
        VehicleCrops,
        build_model,
        make_transforms,
        read_rows,
        seed_everything,
    )
    seed_everything(args.seed)
    rows = read_rows(args.train_csv)
    labels = {v: i for i, v in enumerate(sorted({r["vehicle_id"] for r in rows}))}
    config = trainer.model_config(args, labels)
    config["input_hashes"] = {str(path.resolve()): trainer.sha256(path)
                              for path in [args.train_csv, *([args.weights] if args.weights else [])]}
    checkpoint = trainer.load_resume(args.resume)
    if checkpoint is not None:
        config["input_hashes"] = checkpoint["config"]["input_hashes"]
    sources = [HERE / "reid_model.py", HERE / "trainer.py", Path(__file__).resolve()]
    run = trainer.prepare_run(args, config, checkpoint, sources)
    # The classifier/BN head is freshly initialized here, after seeding, exactly as historically.
    model = build_model(config, weights=None if checkpoint else args.weights, checkpoint=checkpoint).cuda()
    train_transform, _, preprocessing = make_transforms(model, args.image_size, args.resize_mode)
    trainer.write_json(run / "preprocessing.json", preprocessing)
    fit_data = VehicleCrops(rows, args.images_dir, train_transform, labels)
    sampler = FitTwoPlusTwoBatches if args.identity_sampler == "fit_2plus2_v1" else IdentityBatches
    trainer.train(run, model, fit_data, rows, sampler, args, config, checkpoint)


if __name__ == "__main__":
    main()
