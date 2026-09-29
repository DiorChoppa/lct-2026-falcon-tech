"""Shared supervised loop for both stages, restored from the historical scripts/run_experiment.py.

Recipe (both stages): P x K identity batches, cross-entropy with label smoothing + batch-hard
triplet on L2-normalized post-BN features, AdamW with separate backbone/head learning rates,
linear warmup then constant LR, global gradient clipping at 5.0, BF16 autocast, CUDA only.
Torch is imported lazily so that `--help` works on machines without it.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STARTED = time.perf_counter()
CHECKPOINT_SECONDS = 600
# Treatment keys a resume may not change; the original runner rejected such continuations.
FROZEN_KEYS = ("model", "num_classes", "class_ids", "image_size", "neck", "global_pool",
               "resize_mode", "triplet_space", "lr", "head_lr", "weight_decay", "triplet_weight",
               "margin", "label_smoothing", "warmup_steps", "batch_size", "samples_per_id", "seed",
               "amp", "workers", "identity_sampler", "input_hashes", "source_hashes")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def log_event(run, **event):
    event = {"timestamp_utc": datetime.now(UTC).isoformat(),
             "invocation_elapsed_seconds": time.perf_counter() - STARTED, **event}
    line = json.dumps(event, allow_nan=False)
    with (run / "trainlog.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def lr_factor(step, warmup_steps):
    """LambdaLR step0 sets optimizer update1: linear warmup, then constant."""
    if step < 0:
        raise ValueError("Scheduler step must be nonnegative")
    return min(1., (step + 1) / max(warmup_steps, 1))


def add_recipe_arguments(parser):
    """Arguments shared by both stages; defaults are the shipped recipe."""
    parser.add_argument("--config", type=Path, help="JSON file whose keys set argument defaults")
    parser.add_argument("--run-dir", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--weights", type=Path, help="Initial backbone (.safetensors)")
    parser.add_argument("--resume", type=Path, help="Checkpoint of the same run to continue")
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--global-pool", choices=["default", "avg", "token"], default="token")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=["center", "stretch"], default="stretch")
    parser.add_argument("--neck", choices=["none", "bn"], default="bn")
    parser.add_argument("--triplet-space", choices=["post_bn", "pre_bn"], default="post_bn")
    parser.add_argument("--epochs", type=int, default=1, help="Total nominal epochs; ceil(N/batch) batches each")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-id", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5, help="Backbone learning rate")
    parser.add_argument("--head-lr", type=float, default=3e-4, help="BN neck and classifier learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--triplet-weight", type=float, default=1.)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Smoke cap on optimizer updates per invocation; zero disables")
    parser.add_argument("--checkpoint-retention", choices=["all", "last"], default="last",
                        help="'last' deletes the previous checkpoint after a newer one is written")


def parse_with_config(parser, argv=None):
    """Apply `--config` JSON as defaults, so explicit command-line flags still win."""
    known, _ = parser.parse_known_args(argv)
    if known.config:
        values = json.loads(known.config.read_text(encoding="utf-8"))
        values = {key: value for key, value in values.items() if not key.startswith("_")}
        destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(values) - destinations)
        if unknown:
            parser.error(f"Unknown keys in {known.config}: {unknown}")
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def require_cuda(amp):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is prohibited for training")
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False


def model_config(args, labels):
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(num_classes=len(labels), class_ids=list(labels), backbone_stage="full", format_version=1)
    return config


def prepare_run(args, config, checkpoint, source_paths):
    """Fresh run: snapshot sources and config. Resume: same run, identical treatment."""
    run = args.run_dir.resolve()
    config["source_hashes"] = {path.name: sha256(path) for path in source_paths}
    if checkpoint is not None:
        for key in FROZEN_KEYS:
            if config.get(key) != checkpoint["config"].get(key):
                raise ValueError(f"Resume changed treatment or source: {key}; start a new run instead")
        if not (run / "config.json").exists():
            raise FileNotFoundError("Resume must use the original run directory")
        return run
    if run.exists():
        raise FileExistsError("A new run requires a fresh --run-dir; use --resume to continue")
    (run / "source").mkdir(parents=True)
    for path in source_paths:
        shutil.copy2(path, run / "source" / path.name)
    write_json(run / "config.json", config)
    return run


def save_checkpoint(run, model, optimizer, scheduler, config, epoch, batch_cursor, global_step,
                    training_seconds, previous):
    import numpy as np
    import torch
    directory = run / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"epoch{epoch:04d}-batch{batch_cursor:04d}-step{global_step:07d}.pt"
    if not path.exists():
        state = {"model": model.state_dict(), "config": config, "epoch": epoch, "batch_cursor": batch_cursor,
                 "global_step": global_step, "training_seconds": training_seconds,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                 "amp_scaler": None}
        temporary = path.with_suffix(".pt.partial")
        torch.save(state, temporary)
        temporary.replace(path)
    reference = {"path": str(path), "sha256": sha256(path), "epoch": epoch,
                 "batch_cursor": batch_cursor, "global_step": global_step}
    write_json(run / "last.json", reference)
    if (config["checkpoint_retention"] == "last" and previous is not None
            and Path(previous["path"]) != path and Path(previous["path"]).parent == directory):
        Path(previous["path"]).unlink(missing_ok=True)
    return reference


def train(run, model, fit_data, rows, sampler_class, args, config, checkpoint):
    """Run the restored optimization loop until `args.epochs` nominal epochs are complete."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from train.reid_model import batch_hard_triplet

    groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": [p for module in (model.neck, model.classifier) for p in module.parameters()
                          if p.requires_grad], "lr": args.head_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args.warmup_steps))
    resumed = bool(checkpoint)
    epoch = checkpoint["epoch"] if resumed else 0
    cursor = checkpoint["batch_cursor"] if resumed else 0
    step = checkpoint["global_step"] if resumed else 0
    training_seconds = checkpoint["training_seconds"] if resumed else 0.
    if resumed:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        checkpoint.clear()  # Model/optimizer/RNG now own the state; release the CPU copy (~3.6 GB).
    last = json.loads((run / "last.json").read_text(encoding="utf-8")) if resumed else None
    first_step = step
    torch.cuda.reset_peak_memory_stats()
    log_event(run, event="start", gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
              rows=len(rows), classes=config["num_classes"], resumed_step=step)
    smoke_end = False
    while epoch < args.epochs and not smoke_end:
        sampler = sampler_class(rows, args.batch_size, args.samples_per_id, args.seed, epoch)
        loader = DataLoader(fit_data, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        epoch_start, time_start = time.perf_counter(), training_seconds
        last_heartbeat = last_save = time.perf_counter()
        if cursor:
            if args.workers:
                # Worker augmentation streams restart from the fixed epoch seed: replay the prefix.
                loader = iter(loader)
                for _ in range(cursor):
                    next(loader)
            else:
                remaining = list(iter(sampler))[cursor:]
                loader = DataLoader(fit_data, batch_sampler=remaining, num_workers=0, pin_memory=True,
                                    generator=torch.Generator().manual_seed(args.seed + epoch))
        for images, target in loader:
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                features, logits = model.training_outputs(images)
                ce = F.cross_entropy(logits.float(), target, label_smoothing=args.label_smoothing)
                triplet = batch_hard_triplet(features, target, args.margin)
                loss = ce + args.triplet_weight * triplet
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            applied_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            step, cursor = step + 1, cursor + 1
            training_seconds = time_start + time.perf_counter() - epoch_start
            if step == first_step + 1 or step % 20 == 0 or time.perf_counter() - last_heartbeat >= 60:
                log_event(run, event="train", epoch=epoch, batch=cursor, step=step, loss=float(loss.detach()),
                          ce=float(ce.detach()), triplet=float(triplet.detach()), grad_norm=float(grad_norm),
                          applied_lrs=applied_lrs, training_seconds=training_seconds,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          examples_seen=step * args.batch_size)
                last_heartbeat = time.perf_counter()
            if time.perf_counter() - last_save >= CHECKPOINT_SECONDS:
                last = save_checkpoint(run, model, optimizer, scheduler, config, epoch, cursor, step,
                                       training_seconds, last)
                last_save = time.perf_counter()
            smoke_end = args.max_train_batches > 0 and step - first_step >= args.max_train_batches
            if smoke_end:
                break
        if cursor == len(sampler):
            epoch, cursor = epoch + 1, 0
        last = save_checkpoint(run, model, optimizer, scheduler, config, epoch, cursor, step, training_seconds, last)
        log_event(run, event="epoch_end", epoch=epoch, partial=cursor != 0, step=step,
                  training_seconds=training_seconds, checkpoint=last)
    log_event(run, event="complete", epochs_completed=epoch, partial_epoch_batches=cursor, global_step=step,
              training_seconds=training_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return last


def load_resume(path):
    if path is None:
        return None
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def new_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
