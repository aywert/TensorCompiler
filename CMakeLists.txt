cmake_minimum_required(VERSION 3.16)
project(tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Ищем Protobuf
find_package(Protobuf REQUIRED)

# 2. Указываем путь к файлу (рекомендую переименовать .proto3 в .proto для красоты имен)
set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.proto")

# 3. Генерируем файлы. Результат будет в ONNX_SRCS и ONNX_HDRS
protobuf_generate_cpp(ONNX_SRCS ONNX_HDRS ${ONNX_PROTO})

# 4. Создаем ОДИН исполняемый файл
# Включаем в него ваш main.cpp и СГЕНЕРИРОВАННЫЕ исходники
add_executable(${PROJECT_NAME} src/main.cpp ${ONNX_SRCS})

# 5. Настраиваем пути и линковку
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}  # Здесь лежат сгенерированные .pb.h
    ${Protobuf_INCLUDE_DIRS}
    include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Protobuf_LIBRARIES}
)
