"""Сервисный граф: models/model.onnx лидера + второй выход `patches` для сверки по деталям.

Граф жюри не меняется (нативный экстрактор ml/inference требует ровно один выход и
сверяет SHA). Производный граф собирается при сборке образа inference
(deploy/rust.Dockerfile, стадия models) и в git не хранится.

`patches` — выход финального LayerNorm бэкбона (из него же CLS идёт в BN-neck и
`embeddings`) без CLS и register-токенов, L2 по каждому токену: [N, grid_h*grid_w, D],
float32. Глобальный эмбеддинг бит-в-бит тот же, галерея переимпорта не требует.

    python deploy/service_model.py models/model.json /out
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Тензор, из которого граф лидера берёт CLS (Gather index 0 по оси токенов).
TOKENS = "/backbone/norm/LayerNormalization_output_cast_0"
PATCH = 16


def main(manifest_path: Path, out_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    graph_path = manifest_path.parent / manifest["file"]
    with graph_path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != manifest["sha256"]:
        sys.exit(f"model checksum mismatch: {graph_path}")
    model = onnx.load(graph_path)
    graph = model.graph
    if [o.name for o in graph.output] != [manifest["output_name"]]:
        sys.exit(f"ожидался один выход {manifest['output_name']}, есть {[o.name for o in graph.output]}")
    if not any(TOKENS in n.output for n in graph.node):
        sys.exit(f"в графе нет {TOKENS}: сменилась архитектура экспорта, обновить скрипт")

    grid_h, grid_w = manifest["input_height"] // PATCH, manifest["input_width"] // PATCH
    n_patches, dim = grid_h * grid_w, manifest["dim"]
    # Патчи — последние grid_h*grid_w токенов (перед ними CLS и register-токены).
    graph.initializer.extend([
        numpy_helper.from_array(np.array([-n_patches], dtype=np.int64), "patches_start"),
        numpy_helper.from_array(np.array([np.iinfo(np.int64).max], dtype=np.int64), "patches_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "patches_axis"),
    ])
    graph.node.extend([
        helper.make_node("Slice", [TOKENS, "patches_start", "patches_end", "patches_axis"],
                         ["patches_raw"], name="service/patches_slice"),
        helper.make_node("LpNormalization", ["patches_raw"], ["patches"], axis=-1, p=2,
                         name="service/patches_l2"),
    ])
    graph.output.append(helper.make_tensor_value_info(
        "patches", TensorProto.FLOAT, ["batch", n_patches, dim]))
    onnx.checker.check_model(model, full_check=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / manifest["file"]
    onnx.save(model, out_model)
    manifest.update({
        "sha256": hashlib.sha256(out_model.read_bytes()).hexdigest(),
        "derived_from_sha256": manifest["sha256"],
        "patches_output_name": "patches",
        "patch_grid_h": grid_h,
        "patch_grid_w": grid_w,
        "patch_dim": dim,
    })
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{out_model}: patches [N, {n_patches}, {dim}], sha256 {manifest['sha256'][:12]}…")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
