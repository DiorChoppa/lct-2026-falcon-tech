"""Continue external pretraining on public VERI-Wild TRAIN + TEST (organizer Q46).

Reconstructs the fixed wild_tt_e1d recipe, including reused classifier rows and
AdamW moments. This external TEST split is unrelated to the contest jury test.
No contest image is used for fitting; its file hashes are admission references.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train import trainer
from train.stage1_veriwild import parse_train_list


def external_rows(metadata, train_images, test_images):
    rows = parse_train_list((metadata / 'train_list_start0.txt').read_text(encoding='utf-8-sig'))
    for row in rows:
        row['path'] = str(train_images / row['path'])
    test = {}
    for name in ('test_10000_id_query.txt', 'test_10000_id.txt'):
        for line in (metadata / name).read_text().splitlines():
            path, identity, camera = line.split()
            if Path(path).is_absolute() or '..' in Path(path).parts or len(Path(path).parts) != 2:
                raise ValueError('Unsafe external image path')
            if int(Path(path).parts[0]) != int(identity):
                raise ValueError('External identity/path mismatch')
            value = (identity, camera)
            if path in test and test[path] != value:
                raise ValueError('Conflicting external metadata')
            test[path] = value
    rows += [{'path': str(test_images / path), 'vehicle_id': 'veriwild_test:' + str(int(test[path][0])),
              'camera_id': test[path][1]} for path in sorted(test)]
    ids = sorted({row['vehicle_id'] for row in rows})
    labels = {identity: index for index, identity in enumerate(ids)}
    for row in rows:
        row['label'] = labels[row['vehicle_id']]
    return rows, ids


def expand_classifier_moments(optimizer_state, new_rows, old_rows, classes):
    """Keep E2 AdamW history; new external identities receive zero moments."""
    import torch
    key = optimizer_state['param_groups'][1]['params'][-1]
    for name in ('exp_avg', 'exp_avg_sq'):
        old = optimizer_state['state'][key][name]
        expanded = torch.zeros(classes, old.shape[1], dtype=old.dtype)
        expanded[new_rows] = old[old_rows]
        optimizer_state['state'][key][name] = expanded
    return optimizer_state


def admit(rows, contest_images, run):
    reference_paths = sorted(contest_images.glob('*.jpg'))
    if len(reference_paths) != 11416:
        raise ValueError('Admission requires all 11,416 contest images as hash references')
    with ThreadPoolExecutor(max_workers=8) as pool:
        references = set(pool.map(trainer.sha256, reference_paths))
        matches, inventory = [], hashlib.sha256()
        for row, digest in zip(rows, pool.map(trainer.sha256, [Path(r['path']) for r in rows])):
            if digest in references:
                matches.append(row['path'])
            relative = '/'.join(Path(row['path']).parts[-2:])
            inventory.update((json.dumps([relative, row['vehicle_id'], row['camera_id'], digest], separators=(',', ':')) + '\n').encode())
    receipt = {'images': len(rows), 'contest_images': len(reference_paths), 'exact_matches': len(matches),
               'matches': matches, 'inventory_sha256': inventory.hexdigest(),
               'scope': 'Encoded-file hashes, not a perceptual duplicate guarantee'}
    trainer.write_json(run / 'external-admission.json', receipt)
    if matches:
        raise ValueError('External images overlap contest data')


def check_health(run, launch=False):
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.limit', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True)
    temperature, power = map(float, result.stdout.strip().splitlines()[0].split(','))
    if sys.platform == 'win32':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
                        ('total_physical', 'available_physical', 'total_commit', 'available_commit', 'total_virtual', 'available_virtual', 'extra')]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        available = memory.available_physical
    else:
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(info['MemAvailable'].split()[0]) * 1024
    if temperature >= 83 or power != 350 or shutil.disk_usage(run).free < 40 * 2**30 or available < (8 if launch else 2) * 2**30:
        raise RuntimeError('GPU temperature/power, RAM, or disk budget violated')
    return temperature


class Images:
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(row['path']) as image:
            return self.transform(image.convert('RGB')), row['label']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--train-images', type=Path, required=True)
    parser.add_argument('--test-images', type=Path, required=True)
    parser.add_argument('--init-backbone', type=Path)
    parser.add_argument('--init-head', type=Path, help='Full terminal E2 checkpoint, including optimizer')
    parser.add_argument('--contest-images', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--max-minutes', type=float, default=160)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    rows, ids = external_rows(args.metadata_dir, args.train_images, args.test_images)
    if (len(rows), len(ids)) != (416314, 40671):
        raise ValueError('Expected the complete public TRAIN + TEST corpus: 416,314 images / 40,671 IDs')
    if args.dry_run:
        print(json.dumps({'images': len(rows), 'identities': len(ids), 'steps': math.ceil(len(rows) / 16),
                          'note': 'Metadata validation only; a fitting launch also hashes every input'}))
        return
    if not all((args.init_backbone, args.init_head, args.contest_images, args.run_dir)):
        parser.error('Fitting requires both E2 artifacts, contest hash references, and a fresh run directory')
    if not 0 < args.max_minutes <= 240:
        parser.error('--max-minutes must be within (0, 240]')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    admit(rows, args.contest_images, args.run_dir)
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from train.reid_model import (
        IdentityBatches,
        batch_hard_triplet,
        build_model,
        make_transforms,
        seed_everything,
    )
    trainer.require_cuda('bf16')
    check_health(args.run_dir, launch=True)
    seed_everything(20260923)
    config = {'model': 'vit_large_patch16_dinov3.lvd1689m', 'image_size': 256, 'neck': 'bn',
              'global_pool': 'token', 'triplet_space': 'post_bn', 'resize_mode': 'stretch', 'num_classes': len(ids),
              'class_ids': ids, 'seed': 20260923, 'epochs': 1, 'batch_size': 16, 'lr': 3e-5, 'head_lr': 3e-4,
              'warmup_steps': 50, 'workers': 2, 'init_sha256': trainer.sha256(args.init_backbone),
              'init_head_sha256': trainer.sha256(args.init_head), 'source_sha256': trainer.sha256(Path(__file__))}
    config['input_hashes'] = {name: trainer.sha256(args.metadata_dir / name) for name in
                              ('train_list_start0.txt', 'test_10000_id_query.txt', 'test_10000_id.txt')}
    config['source_hashes'] = {name: trainer.sha256(Path(__file__).with_name(name)) for name in
                               ('stage1_tt.py', 'stage1_veriwild.py', 'reid_model.py', 'trainer.py')}
    trainer.write_json(args.run_dir / 'config.json', config)
    model = build_model(config, weights=args.init_backbone)
    head = torch.load(args.init_head, map_location='cpu', weights_only=False, mmap=True)
    old_ids = {identity: index for index, identity in enumerate(head['config']['class_ids'])}
    reused = [(index, old_ids[identity]) for index, identity in enumerate(ids) if identity in old_ids]
    if len(reused) != 30671:
        raise ValueError('E2 checkpoint must contain all 30,671 namespaced TRAIN identities')
    new_rows, old_rows = [n for n, _ in reused], [o for _, o in reused]
    model.neck.load_state_dict({key[5:]: value for key, value in head['model'].items() if key.startswith('neck.')})
    model.classifier.weight.data[new_rows] = head['model']['classifier.weight'][old_rows].float()
    moments = expand_classifier_moments(head['optimizer'], new_rows, old_rows, len(ids))
    del head
    model = model.cuda()
    transform, _, _ = make_transforms(model, 256, 'stretch')
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 3e-5},
                                  {'params': [p for m in (model.neck, model.classifier) for p in m.parameters()], 'lr': 3e-4}], weight_decay=.05)
    optimizer.load_state_dict(moments)
    for group, lr in zip(optimizer.param_groups, (3e-5, 3e-4)):
        group['lr'], group['initial_lr'] = lr, lr
    del moments
    gc.collect()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: trainer.lr_factor(step, 50))
    sampler = IdentityBatches(rows, 16, 4, 20260923, 0)
    loader = DataLoader(Images(rows, transform), batch_sampler=sampler, num_workers=2, pin_memory=False)
    started, completed = time.monotonic(), 0
    model.train()
    for images, target in loader:
        if (args.run_dir / 'STOP').exists() or time.monotonic() - started >= args.max_minutes * 60:
            break
        if completed % 100 == 0:
            check_health(args.run_dir)
        images, target = images.cuda(), target.cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features, logits = model.training_outputs(images)
            loss = F.cross_entropy(logits.float(), target, label_smoothing=.1) + batch_hard_triplet(features, target, .3)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite external loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        completed += 1
        if completed % 20 == 0:
            trainer.log_event(args.run_dir, event='train', step=completed, loss=float(loss.detach()))
    save_file({key: value.detach().cpu().contiguous() for key, value in model.backbone.state_dict().items()}, str(args.run_dir / 'backbone.safetensors'))
    report = {'complete': completed == 26020, 'steps': completed, 'minutes': (time.monotonic() - started) / 60,
              'backbone_sha256': trainer.sha256(args.run_dir / 'backbone.safetensors')}
    trainer.write_json(args.run_dir / 'completion.json', report)
    if not report['complete']:
        raise RuntimeError('Incomplete external continuation; do not promote its backbone')


if __name__ == '__main__':
    main()
