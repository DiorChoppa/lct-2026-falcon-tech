"""Offline image/BBox encoder and the small training-only identity head."""
import csv
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import timm
from torchvision import transforms as T
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from safetensors.torch import load_file


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty image manifest: {path}")
    required = {"image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing manifest columns: {required - rows[0].keys()}")
    return rows


class VehicleCrops(Dataset):
    def __init__(self, rows, images_dir, transform, labels=None, training_masks=None, crop_cache_dir=None):
        self.rows, self.images_dir, self.transform = rows, Path(images_dir), transform
        self.labels = labels or {v: i for i, v in enumerate(sorted({r["vehicle_id"] for r in rows}))}
        self.training_masks = training_masks or {}
        self.crop_cache_dir = Path(crop_cache_dir) if crop_cache_dir else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
        if self.crop_cache_dir is not None:
            cached = np.load(self.crop_cache_dir / (row["image_id"] + ".npy"), allow_pickle=False)
            if cached.dtype != np.uint8 or cached.shape != (h, w, 3):
                raise ValueError(f"Invalid cached fitting crop: {row['image_id']}")
            # Verified cache already includes the fitting-only reviewed mask; do not apply it twice.
            return self.transform(Image.fromarray(cached)), self.labels[row["vehicle_id"]]
        with Image.open(self.images_dir / (row["image_id"] + ".jpg")) as image:
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image.width or y + h > image.height:
                raise ValueError(f"Invalid BBox for {row['image_id']}")
            image = image.convert("RGB")
            if row["image_id"] in self.training_masks:
                mx, my, mw, mh = self.training_masks[row["image_id"]]
                image.paste((0, 0, 0), (mx, my, mx + mw, my + mh))
            crop = image.crop((x, y, x + w, y + h))
        return self.transform(crop), self.labels[row["vehicle_id"]]


class IdentityBatches(Sampler):
    """Seeded P x K draws: unique identities, distinct cameras first, no model metadata."""
    def __init__(self, rows, batch_size, samples_per_id, seed, epoch=0):
        if samples_per_id < 2 or batch_size % samples_per_id:
            raise ValueError("batch-size must be a multiple of samples-per-id >= 2")
        self.p, self.k = batch_size // samples_per_id, samples_per_id
        self.groups = {}
        for index, row in enumerate(rows):
            self.groups.setdefault(row["vehicle_id"], {}).setdefault(row["camera_id"], []).append(index)
        if self.p < 2 or self.p > len(self.groups):
            raise ValueError("Each triplet batch requires at least two distinct identities")
        self.steps = math.ceil(len(rows) / batch_size)
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000003)
        for _ in range(self.steps):
            batch = []
            for identity in rng.sample(sorted(self.groups), self.p):
                cameras = self.groups[identity]
                order = list(cameras)
                rng.shuffle(order)
                chosen = [rng.choice(cameras[c]) for c in order[:self.k]]
                remaining = [i for indices in cameras.values() for i in indices if i not in chosen]
                rng.shuffle(remaining)
                chosen += remaining[:self.k - len(chosen)]
                while len(chosen) < self.k:
                    chosen.append(rng.choice([i for indices in cameras.values() for i in indices]))
                batch.extend(chosen)
            yield batch


class FitTwoPlusTwoBatches(IdentityBatches):
    """One FIT camera-balance replacement; preserve parent sampler RNG draws."""
    def __init__(self, rows, batch_size, samples_per_id, seed, epoch=0):
        super().__init__(rows, batch_size, samples_per_id, seed, epoch)
        if self.k != 4:
            raise ValueError("fit_2plus2_v1 requires K=4")
        self.rows = rows
        self.eligible = {identity for identity, cameras in self.groups.items()
                         if len(cameras) == 2 and min(map(len, cameras.values())) >= 2}

    def __iter__(self):
        from collections import Counter
        for batch_number, batch in enumerate(super().__iter__()):
            result = batch[:]
            for block_number in range(self.p):
                offset = block_number * self.k
                block = batch[offset:offset + self.k]
                identity = self.rows[block[0]]["vehicle_id"]
                counts = Counter(self.rows[index]["camera_id"] for index in block)
                if identity not in self.eligible or sorted(counts.values()) != [1, 3]:
                    continue
                majority, minority = max(counts, key=counts.get), min(counts, key=counts.get)
                rng = random.Random(f'{self.seed}/{self.epoch}/{batch_number}/{block_number}/2plus2')
                position = rng.choice([j for j, index in enumerate(block)
                                       if self.rows[index]["camera_id"] == majority])
                replacement = rng.choice([index for index in self.groups[identity][minority]
                                          if index not in block])
                result[offset + position] = replacement
            yield result


def validate_backbone_stage(config, previous=None):
    stage = config.get("backbone_stage", "full")
    if stage not in ("full", "stage3", "shared_stage34"):
        raise ValueError("Unknown backbone stage")
    if stage == "stage3" and (config["model"] != "convnext_base.dinov3_lvd1689m" or
                             config.get("global_pool", "default") != "default"):
        raise ValueError("Stage3 is verified only for DINOv3 ConvNeXt-Base with default pooling")
    if stage == "shared_stage34" and (config["model"] != "convnext_base.dinov3_lvd1689m"
            or config.get("global_pool", "default") != "default" or config.get("neck") != "bn"
            or config.get("triplet_space", "post_bn") != "post_bn"):
        raise ValueError("Shared Stage3/4 requires exact Base, default pooling, BN and post-BN metric")
    if previous is not None and stage != previous.get("backbone_stage", "full"):
        raise ValueError("Resume changed backbone stage; start a separately declared branch")


class ConvNeXtStage3(nn.Module):
    """Truncated global descriptor: pretrained stem/stages0..2, GAP, fresh LN."""
    def __init__(self, full):
        super().__init__()
        if full.num_features != 1024 or len(full.stages) != 4 or full.feature_info[2]["num_chs"] != 512:
            raise ValueError("Unexpected ConvNeXt-Base layout")
        self.stem = full.stem
        self.stages = nn.Sequential(*list(full.stages.children())[:3])
        self.num_features = 512
        self.norm = nn.LayerNorm(512, eps=1e-6)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        self.pretrained_cfg = dict(full.pretrained_cfg)
        self.feature_info = list(full.feature_info[:3])

    def forward_features(self, images):
        return self.stages(self.stem(images))

    def forward(self, images):
        return self.norm(self.forward_features(images).mean(dim=(2, 3)))


class ConvNeXtSharedStage(nn.Module):
    """One full pretrained trunk, pooled Stage3+Stage4; no regional matching."""
    def __init__(self, full):
        super().__init__()
        if full.num_features != 1024 or len(full.stages) != 4 or full.feature_info[2]["num_chs"] != 512:
            raise ValueError("Unexpected ConvNeXt-Base layout")
        self.stem, self.stages = full.stem, full.stages
        self.norm_pre, self.head = full.norm_pre, full.head
        self.stage3_norm = nn.LayerNorm(512, eps=1e-6)
        nn.init.ones_(self.stage3_norm.weight)
        nn.init.zeros_(self.stage3_norm.bias)
        self.num_features = 1536
        self.pretrained_cfg = dict(full.pretrained_cfg)
        self.feature_info = list(full.feature_info)

    def forward(self, images):
        x = self.stem(images)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        stage3 = self.stage3_norm(x.mean(dim=(2, 3)))
        stage4 = self.head(self.norm_pre(self.stages[3](x)))
        return torch.cat((stage3, stage4), dim=1)


class ReIDModel(nn.Module):
    def __init__(self, name, classes, image_size=224, neck="bn", global_pool="default", triplet_space="post_bn"):
        super().__init__()
        if triplet_space not in ("post_bn", "pre_bn"):
            raise ValueError("Unknown triplet feature space")
        self.triplet_space = triplet_space
        kwargs = {"img_size": image_size} if "vit" in name else {}
        if global_pool != "default":
            if name not in ("vit_small_patch16_dinov3.lvd1689m", "vit_base_patch16_dinov3.lvd1689m", "vit_large_patch16_dinov3.lvd1689m") or global_pool not in ("avg", "token"):
                raise ValueError("Alternate pooling is supported only for the declared DINOv3 ViT-S, ViT-B and ViT-L variants")
            kwargs["global_pool"] = global_pool
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0, **kwargs)
        dim = self.backbone.num_features
        self.neck = nn.BatchNorm1d(dim) if neck == "bn" else nn.Identity()
        if neck == "bn":
            self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images):
        # The classifier is deliberately absent from the inference graph.
        return self.normalize_metric(self.neck(self.backbone(images)))

    def normalize_metric(self, features):
        features = features.float()
        if isinstance(self.backbone, ConvNeXtSharedStage):
            features = torch.cat((F.normalize(features[:, :512], dim=1),
                                  F.normalize(features[:, 512:], dim=1)), dim=1)
        return F.normalize(features, dim=1)

    def share_stages34(self):
        classes = self.classifier.out_features
        self.backbone = ConvNeXtSharedStage(self.backbone)
        self.neck = nn.BatchNorm1d(1536)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(1536, classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def truncate_stage3(self):
        classes = self.classifier.out_features
        self.backbone = ConvNeXtStage3(self.backbone)
        if isinstance(self.neck, nn.BatchNorm1d):
            self.neck = nn.BatchNorm1d(512)
            self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def training_outputs(self, images):
        raw = self.backbone(images)
        features = self.neck(raw)
        metric_features = raw if self.triplet_space == "pre_bn" else features
        return self.normalize_metric(metric_features), self.classifier(features)


def build_model(config, weights=None, checkpoint=None):
    validate_backbone_stage(config, checkpoint.get("config", {}) if checkpoint is not None else None)
    stage3 = config.get("backbone_stage", "full") == "stage3"
    shared = config.get("backbone_stage", "full") == "shared_stage34"
    if shared and checkpoint is None and not weights:
        raise ValueError("Shared readout requires full pretrained weights or strict shared checkpoint")
    if stage3 and checkpoint is None and not weights:
        raise ValueError("Stage3 requires full pretrained weights or a strict stage3 checkpoint")
    model = ReIDModel(config["model"], config["num_classes"], config["image_size"], config["neck"], config.get("global_pool", "default"), config.get("triplet_space", "post_bn"))
    if checkpoint is not None:
        if stage3:
            model.truncate_stage3()
        elif shared:
            model.share_stages34()
        model.load_state_dict(checkpoint["model"], strict=True)
    elif weights:
        path = Path(weights)
        state = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state, strict=True)
        if stage3:
            model.truncate_stage3()
        elif shared:
            model.share_stages34()
    return model


def make_transforms(model, image_size, resize_mode="center"):
    config = timm.data.resolve_model_data_config(model.backbone)
    config["input_size"] = (3, image_size, image_size)
    evaluation = timm.data.create_transform(**config, is_training=False)
    training = timm.data.create_transform(**config, is_training=True, scale=(0.8, 1.0),
                                          ratio=(0.75, 1.3333333), hflip=0.5,
                                          color_jitter=0.1, re_prob=0.1)
    if resize_mode == "stretch":
        resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        normalize = T.Normalize(config["mean"], config["std"])
        evaluation = T.Compose([resize, T.ToTensor(), normalize])
        training = T.Compose([resize, T.RandomHorizontalFlip(0.5), T.ColorJitter(0.1, 0.1, 0.1),
                              T.ToTensor(), normalize, T.RandomErasing(p=0.1)])
    elif resize_mode != "center":
        raise ValueError("Unsupported training resize mode")
    config["resize_mode"] = resize_mode
    return training, evaluation, config


def batch_hard_triplet(features, labels, margin):
    distance = torch.cdist(features.float(), features.float(), p=2)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = labels[:, None].ne(labels[None, :])
    valid = same.any(1) & different.any(1)
    if not bool(valid.all()):
        raise ValueError("Every anchor needs a positive and a negative in its batch")
    positive = distance.masked_fill(~same, -torch.inf).max(1).values
    negative = distance.masked_fill(~different, torch.inf).min(1).values
    return F.relu(positive - negative + margin).mean()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def self_check():
    rows = [{"vehicle_id": str(i), "camera_id": str(c)} for i in range(3) for c in range(3)]
    batches = list(IdentityBatches(rows, 8, 4, 7))
    assert batches == list(IdentityBatches(rows, 8, 4, 7))
    for batch in batches:
        identities = {rows[i]["vehicle_id"] for i in batch}
        assert len(identities) == 2 and len(batch) == 8
        assert all(len({rows[i]["camera_id"] for i in batch if rows[i]["vehicle_id"] == v}) == 3 for v in identities)
    features = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]], requires_grad=True)
    loss = batch_hard_triplet(features, torch.tensor([0, 0, 1, 1]), 0.3)
    assert loss.item() == 0
    loss.backward()
    assert bool(torch.isfinite(features.grad).all())
    print("CPU sampler and batch-hard loss checks passed")


if __name__ == "__main__":
    self_check()
