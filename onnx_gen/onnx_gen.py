import onnx
import numpy as np
from onnx import helper, TensorProto

# 1. Создаем Вход графа (Placeholder для ваших будущих данных)
# Аргументы: Тип данных, Shape (None или 'batch' для динамического размера)
X = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [1, 2])

# 2. Создаем Инициализаторы (Веса и Смещение)
# Веса матрицы (2x2)
weights_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
W = helper.make_tensor(
    name='weights',
    data_type=TensorProto.FLOAT,
    dims=[2, 2],
    vals=weights_data.tobytes(),
    raw=True  # Попадет в raw_data
)

# Смещение (Bias)
bias_data = np.array([0.5, 0.5], dtype=np.float32)
B = helper.make_tensor(
    name='bias',
    data_type=TensorProto.FLOAT,
    dims=[2],
    vals=bias_data.tolist(),
    raw=False # Попадет в float_data для проверки вашего парсера
)

# 3. Создаем Выход графа
Y = helper.make_tensor_value_info('output_tensor', TensorProto.FLOAT, [1, 2])

# 4. Создаем Узлы (Nodes)
# Операция Gemm (General Matrix Multiplication)
# Входы ноды: ['input_tensor', 'weights', 'bias']
# Выход ноды: ['output_tensor']
node_gemm = helper.make_node(
    'Gemm',                  # Тип операции
    ['input_tensor', 'weights', 'bias'], # Имена входов
    ['output_tensor'],       # Имя выхода
    name='my_gemm_layer',    # Имя самой ноды
    alpha=1.0,               # Атрибуты операции
    beta=1.0,
    transB=1                 # Транспонировать веса (часто для Gemm)
)

# Добавим еще одну операцию для цепочки - ReLU (активация)
# Она возьмет выход Gemm и выдаст финальный результат
Y_final = helper.make_tensor_value_info('final_result', TensorProto.FLOAT, [1, 2])
node_relu = helper.make_node(
    'Relu',
    ['output_tensor'],
    ['final_result']
)

# 5. Собираем Граф
graph = helper.make_graph(
    [node_gemm, node_relu],  # Список узлов
    'simple_linear_model',   # Имя графа
    [X],                     # Входы
    [Y_final],               # Выходы
    [W, B]                   # Инициализаторы
)

# 6. Сохраняем модель
model = helper.make_model(graph, producer_name='my_parser_test')
onnx.save(model, '../examples/linear_model.onnx')

print("Модель linear_model.onnx успешно создана!")
