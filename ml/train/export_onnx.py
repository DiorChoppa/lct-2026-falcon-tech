"""Export a stage-2 checkpoint to the shipped mixed FP16/FP32 ONNX encoder, on CPU.

Restored pipeline of the shipped graph (models/model.onnx):
1. scripts/export_encoder.py: FP32 export, opset 17, legacy exporter, DINOv3 RoPE frozen for
   the fixed 256x256 grid, float32 input `images` [batch,3,256,256] -> `embeddings`.
2. artifacts/vitl_2plus2_single_native_v1/run.py: strict ONNX shape inference fixes the
   output to [batch,1024] (historical FP32 graph sha256 4a509e53...).
3. artifacts/vitl_2plus2_single_native_v1/mixed/convert.py + residual.py: onnxruntime's
   float16 converter; LayerNorm/Softmax/reductions/BN, LayerScale residual stream and the
   final L2 normalization stay FP32 (historical result sha256 ff7194b1... = shipped model).
The training classifier is not part of the exported graph.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from graphlib import TopologicalSorter
from pathlib import Path

SHIPPED_SHA256 = "a1a63bcbfc00388ad6612a33c4861ffae8684a67bb3c0f1cb0d380b0b4bec1e0"
HISTORICAL_FP32_SHA256 = "4a509e537c949a1bf19042ff0e5956239d60a104ddcb714d6882f977b4061f02"
HISTORICAL_CHECKPOINT_SHA256 = "1fb7d811ee5b987138b5ba223ea97d299628de9addfe83bfc205dc827a490696"
CONVERTER_SHA256 = "9cc4d882196728b7ea9e70159ec51808a130b7f9e470f7b74fd2487e0e149f70"  # ORT 1.24.4, CRLF
SENSITIVE = ["LayerNormalization", "Softmax", "ReduceL2", "ReduceMean", "ReduceSum", "ReduceSumSquare",
             "BatchNormalization"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# --- FP32 export (scripts/export_encoder.py) -------------------------------------------------

def freeze_spatial_rope(model, image_size):
    import torch

    class FixedSpatialRotaryEmbedding(torch.nn.Module):
        """Export-only exact RoPE values for fixed H/W; batch remains dynamic."""
        def __init__(self, embedding):
            super().__init__()
            self.register_buffer("embedding", embedding.detach().clone())

        def get_embed(self, shape=None):
            return self.embedding

    backbone = model.backbone
    rope = getattr(backbone, "rope", None)
    if type(rope).__name__ != "RotaryEmbeddingDinoV3" or backbone.training:
        raise ValueError("Static RoPE requires an eval-mode DINOv3 rotary encoder")
    patch = backbone.patch_embed
    if tuple(patch.img_size) != (image_size, image_size):
        raise ValueError("Configured patch embedding size differs from export")
    with torch.inference_mode():
        embedding = rope.get_embed(shape=patch.grid_size)
    backbone.rope = FixedSpatialRotaryEmbedding(embedding)
    return {"grid_size": list(patch.grid_size), "dtype": str(embedding.dtype), "shape": list(embedding.shape)}


def check_float32_io(graph):
    import onnx
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("Export must have exactly one image input and one embedding output")
    if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for value in [*graph.graph.input, *graph.graph.output]):
        raise ValueError("External ONNX input/output must remain float32")


def dimensions(value):
    return [item.dim_param if item.dim_param else item.dim_value for item in value.type.tensor_type.shape.dim]


def fix_shapes(graph):
    """Strict shape inference; nodes and initializers must be unchanged (run.py shape proof)."""
    import onnx
    inferred = onnx.shape_inference.infer_shapes(graph, check_type=True, strict_mode=True, data_prop=True)
    if (list(inferred.graph.node) != list(graph.graph.node)
            or list(inferred.graph.initializer) != list(graph.graph.initializer)):
        raise ValueError("Shape inference changed graph nodes or initializers")
    return inferred


def export_fp32(checkpoint, destination):
    import onnx
    import torch

    from train.checkpoint_inference import inference_transform, load_encoder
    torch.set_num_threads(2)
    model, state, module, source = load_encoder(checkpoint)
    config = state["config"]
    del state
    model.cpu().float().eval()
    _, preprocessing = inference_transform(module, model, config, "stretch")
    rope = freeze_spatial_rope(model, config["image_size"])
    size = config["image_size"]
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, size, size), destination, opset_version=17, dynamo=False,
                          input_names=["images"], output_names=["embeddings"],
                          dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
                          external_data=False)
    graph = onnx.load(destination)
    onnx.checker.check_model(graph, full_check=True)
    check_float32_io(graph)
    if any("classifier" in value.name for value in graph.graph.initializer):
        raise ValueError("Training classifier unexpectedly included in inference graph")
    expected = [["batch", 3, size, size], ["batch", model.backbone.num_features]]
    if [dimensions(graph.graph.input[0]), dimensions(graph.graph.output[0])] != expected:
        graph = fix_shapes(graph)
        if [dimensions(graph.graph.input[0]), dimensions(graph.graph.output[0])] != expected:
            raise ValueError("Only batch may be dynamic")
        onnx.checker.check_model(graph, full_check=True)
        onnx.save_model(graph, destination, save_as_external_data=False)
    check_float32_io(graph)
    return {"model": config["model"], "preprocessing": preprocessing, "static_spatial_rope": rope,
            "model_source_sha256": sha256(source), "input": expected[0], "output": expected[1]}


# --- Mixed FP16/FP32 conversion (mixed/convert.py) -----------------------------------------

def load_converter():
    """onnxruntime/transformers/float16.py, loaded by path (NumPy/ONNX only, no ORT session)."""
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("onnxruntime (historically onnxruntime-gpu 1.24.4) is required for conversion")
    path = Path(next(iter(spec.submodule_search_locations))) / "transformers/float16.py"
    # The historical file came from a Windows wheel: compare with CRLF line endings.
    text = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if hashlib.sha256(text).hexdigest() != CONVERTER_SHA256:
        raise ImportError(f"{path} is not the onnxruntime 1.24.4 float16 converter used for the shipped "
                          "graph; run with `uv run --with onnxruntime==1.24.4 ...`")
    module_spec = importlib.util.spec_from_file_location("pinned_ort_float16", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module, CONVERTER_SHA256


def terminal_norm_nodes(graph):
    selected = {n.name for n in graph.node if n.op_type == "ReduceL2"}
    if len(selected) != 1:
        raise ValueError("Expected one final L2 reduction")
    values = {v for n in graph.node if n.name in selected for v in n.output}
    while True:
        added = [n for n in graph.node if n.name not in selected and any(v in values for v in n.input)]
        if not added:
            break
        for node in added:
            if node.op_type not in {"Clip", "Expand", "Div", "Cast", "Identity"}:
                raise ValueError("Unexpected final normalization descendant: " + node.op_type)
            selected.add(node.name)
            values.update(node.output)
    if len(selected) > 8 or not all(v.name in values for v in graph.output):
        raise ValueError("Final normalization must terminate directly at outputs")
    return sorted(selected)


def sort_graph(graph):
    nodes = list(graph.node)
    producers = {v: i for i, n in enumerate(nodes) for v in n.output if v}
    if len(producers) != sum(bool(v) for n in nodes for v in n.output):
        raise ValueError("Duplicate graph value producers")
    order = TopologicalSorter({i: {producers[v] for v in n.input if v in producers}
                               for i, n in enumerate(nodes)}).static_order()
    sorted_nodes = [nodes[i] for i in order]
    del graph.node[:]
    graph.node.extend(sorted_nodes)


def convert_model(model, converter, blocks=24, width=1024, channel=757):
    import onnx

    from train.residual import remove_residual_roundtrips, select_residuals
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute):
        raise ValueError("This candidate supports only the known flat encoder graph")
    names = [n.name for n in model.graph.node]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Unique nonempty source node names required")
    block = sorted(set(converter.DEFAULT_OP_BLOCK_LIST) | set(SENSITIVE))
    selected, rows, scales = select_residuals(model.graph, blocks, width, channel)
    terminal = terminal_norm_nodes(model.graph)
    terminal_rows = [{"node": n.name, "op": n.op_type, "output": n.output[0], "source_inputs": list(n.input),
                      "source_consumers": [c.name for c in model.graph.node if n.output[0] in c.input]}
                     for n in model.graph.node if n.name in terminal]
    nodes = sorted(set(terminal) | set(selected))
    options = {"min_positive_val": 5.96e-8, "max_finite_val": 65504.0, "keep_io_types": True,
               "disable_shape_infer": True, "op_block_list": block, "node_block_list": nodes,
               "force_fp16_initializers": False, "force_fp16_inputs": None,
               "use_bfloat16_as_blocked_nodes_dtype": False}
    model = converter.convert_float_to_float16(model, **options)
    remove_residual_roundtrips(model.graph, rows + terminal_rows, block, nodes)
    sort_graph(model.graph)
    tensors = {t.name: t for t in model.graph.initializer}
    for name, evidence in scales.items():
        tensor = tensors[name]
        if tensor.data_type != onnx.TensorProto.FLOAT or hashlib.sha256(tensor.SerializeToString()).hexdigest() != evidence["source_tensor_sha256"]:
            raise ValueError("Original LayerScale FP32 initializer changed")
    if any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for v in [*model.graph.input, *model.graph.output]):
        raise ValueError("External float32 IO changed")
    types = {v.name: v.type.tensor_type.elem_type for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    types.update({t.name: t.data_type for t in model.graph.initializer})
    floating = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}
    for n in model.graph.node:
        edges = [*n.input, *n.output]
        if (n.op_type in SENSITIVE or n.name in nodes) and any(
                types.get(v) in floating and types[v] != onnx.TensorProto.FLOAT for v in edges):
            raise ValueError("Sensitive operation retains non-FP32 floating edge: " + n.name)
    return model, options


def convert_mixed(source, destination):
    import onnx
    converter, converter_sha = load_converter()
    graph = onnx.load(source, load_external_data=False)
    graph, options = convert_model(graph, converter)
    onnx.checker.check_model(graph, full_check=True)
    counts = dict(Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in graph.graph.initializer))
    if not counts.get("FLOAT16") or not counts.get("FLOAT"):
        raise ValueError("Expected mixed FP16/FP32 initializers")
    onnx.save_model(graph, destination, save_as_external_data=False)
    return {"initializer_types": counts, "converter_sha256_crlf": converter_sha,
            "converter_options": {k: v for k, v in options.items() if k != "node_block_list"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Terminal stage-2 checkpoint (historical: epoch0008-batch0000-step0002864.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Export is a CPU operation, as historically.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint_sha = sha256(args.checkpoint)
    fp32, mixed = args.output / "encoder.fp32.onnx", args.output / "model.onnx"
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
              "checkpoint_matches_historical": checkpoint_sha == HISTORICAL_CHECKPOINT_SHA256}
    report.update(export_fp32(args.checkpoint, fp32))
    report.update(fp32_sha256=sha256(fp32), fp32_matches_historical=sha256(fp32) == HISTORICAL_FP32_SHA256)
    report.update(convert_mixed(fp32, mixed))
    report.update(onnx_sha256=sha256(mixed), onnx_bytes=mixed.stat().st_size,
                  matches_shipped_model=sha256(mixed) == SHIPPED_SHA256, export_seconds=time.perf_counter() - started)
    import onnx
    import torch
    report.update(torch_version=torch.__version__, onnx_version=onnx.__version__)
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
