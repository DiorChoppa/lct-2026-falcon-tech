"""Reconstruct a local checkpoint using its hash-verified archived encoder source."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_encoder(path):
    import torch
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = checkpoint["config"]["source_hashes"]["reid_model.py"]
    candidates = [path.parent.parent / "source/reid_model.py", Path(__file__).with_name("reid_model.py")]
    source = next((p for p in candidates if p.is_file() and digest(p) == expected), None)
    if source is None:
        raise ValueError("Neither archived nor live encoder matches the checkpoint source hash")
    spec = importlib.util.spec_from_file_location("lct_checkpoint_" + expected, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.build_model(checkpoint["config"], checkpoint=checkpoint).eval()
    return model, checkpoint, module, source


class Letterbox:
    def __init__(self, size, mean):
        self.size, self.fill = size, tuple(round(x * 255) for x in mean)

    def __call__(self, image):
        from PIL import Image
        scale = self.size / max(image.size)
        width, height = (max(1, round(x * scale)) for x in image.size)
        resized = image.resize((width, height), Image.Resampling.BICUBIC)
        result = Image.new("RGB", (self.size, self.size), self.fill)
        result.paste(resized, ((self.size - width) // 2, (self.size - height) // 2))
        return result


def inference_transform(module, model, config, mode=None):
    from torchvision import transforms as T
    size = config["image_size"]
    _, center, preprocessing = module.make_transforms(model, size)
    mode = mode or config.get("resize_mode", "center")
    normalize = T.Compose([T.ToTensor(), T.Normalize(preprocessing["mean"], preprocessing["std"])])
    if mode == "center":
        transform = center
    elif mode == "stretch":
        transform = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), normalize])
    elif mode == "letterbox":
        transform = T.Compose([Letterbox(size, preprocessing["mean"]), normalize])
    else:
        raise ValueError("Unknown explicit preprocessing mode")
    return transform, {**preprocessing, "resize_mode": mode}
