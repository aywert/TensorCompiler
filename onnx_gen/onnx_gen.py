import onnx
from onnx import helper, TensorProto
import numpy as np

# Создаем узел с различными атрибутами
test_node = helper.make_node(
    'TestOp',  # Это может быть любой тип операции
    inputs=['input'],
    outputs=['output'],
    int_attr=42,
    float_attr=3.14,
    string_attr="hello",
    ints_attr=[1, 2, 3, 4, 5],
    floats_attr=[1.1, 2.2, 3.3]
)

graph = helper.make_graph(
    [test_node],
    'test_attributes',
    inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT, [1])],
    outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])]
)

model = helper.make_model(graph, producer_name='test_attributes')
onnx.save(model, '../examples/test_attributes.onnx')
print("Модель сохранена как test_attributes.onnx")