import onnx
from onnx import helper, TensorProto
import numpy as np

# 1. Вход графа (например, вектор признаков размера 4)
X = helper.make_tensor_value_info('input_x', TensorProto.FLOAT, [1, 4])

# 2. Инициализаторы (Веса)
# Веса для Gemm (4x4)
w_data = np.random.randn(4, 4).astype(np.float32)
W = helper.make_tensor('weights_1', TensorProto.FLOAT, [4, 4], w_data.tobytes(), raw=True)

# Смещение (4)
b_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
B = helper.make_tensor('bias_1', TensorProto.FLOAT, [4], b_data.tolist(), raw=False)

# Веса для другой ветки (умножение на константу)
scale_data = np.array([2.0], dtype=np.float32)
Scale = helper.make_tensor('scale_factor', TensorProto.FLOAT, [1], scale_data.tolist())

# 3. Узлы (Nodes)
# Нода 1: Gemm (X * W + B)
node1 = helper.make_node(
    'Gemm',
    ['input_x', 'weights_1', 'bias_1'],
    ['gemm_out'],
    name='layer_1_gemm',
    alpha=1.0, beta=1.0, transB=1
)

# Нода 2: ReLU
node2 = helper.make_node(
    'Relu',
    ['gemm_out'],
    ['relu_out'],
    name='layer_2_relu'
)

# Нода 3: Mul (Умножаем вход X на константу Scale)
# Это создает параллельную ветку
node3 = helper.make_node(
    'Mul',
    ['input_x', 'scale_factor'],
    ['parallel_out'],
    name='parallel_branch'
)

# Нода 4: Add (Складываем выход ReLU и параллельную ветку)
# Здесь тензор 'parallel_out' и 'relu_out' встречаются
node4 = helper.make_node(
    'Add',
    ['relu_out', 'parallel_out'],
    ['final_sum'],
    name='merge_node'
)

# 4. Выход графа
Y = helper.make_tensor_value_info('final_sum', TensorProto.FLOAT, [1, 4])

# 5. Сборка
graph = helper.make_graph(
    [node1, node2, node3, node4],
    'ComplexTestGraph',
    [X],
    [Y],
    [W, B, Scale]
)

model = helper.make_model(graph, producer_name='tenc-compiler-test')

# Добавим метаданные для красоты
model.model_version = 1
onnx.save(model, '../examples/complex_model.onnx')

print("Сложная модель 'complex_model.onnx' создана.")
