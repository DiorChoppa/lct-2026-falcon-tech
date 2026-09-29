"""Генератор crates/inference/tests/fixtures/smoke_model.onnx.

Разовый dev-инструмент, не часть CI (в образе rust:1.92, где гоняются
тесты, Python нет). Запускать вручную при необходимости перерегенерировать
фикстуру: `python3 gen_smoke_model.py` из этого каталога. Требует пакет
`onnx` (`pip install onnx`).

Модель детерминированная и не про качество эмбеддингов — только про то,
что весь Rust-пайплайн (decode -> crop -> resize -> normalize -> ORT ->
L2) работает. Вход [N,3,4,4], глобальный average pooling по H,W -> [N,3]
(дальше можно руками посчитать ожидаемый вектор в тесте). Второй выход
"patches" — ReduceMax по тем же осям, тоже [N,3]: не настоящие патч-токены,
только чтобы проверить путь with_patches (два выхода из одной сессии).
"""

import onnx
from onnx import TensorProto, helper

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 4, 4])
embedding_output = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["N", 3])
patches_output = helper.make_tensor_value_info("patches", TensorProto.FLOAT, ["N", 3])

embedding_node = helper.make_node(
    "ReduceMean",
    inputs=["input"],
    outputs=["embedding"],
    axes=[2, 3],
    keepdims=0,
)
patches_node = helper.make_node(
    "ReduceMax",
    inputs=["input"],
    outputs=["patches"],
    axes=[2, 3],
    keepdims=0,
)

graph = helper.make_graph(
    [embedding_node, patches_node],
    "smoke_model",
    [input_tensor],
    [embedding_output, patches_output],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
# IR version 8 — совместимо и со старым, и с новым onnxruntime (локальный
# python-пакет по умолчанию пишет IR 12, который понимает не каждый рантайм).
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "smoke_model.onnx")
print("wrote smoke_model.onnx")
