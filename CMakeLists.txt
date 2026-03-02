cmake_minimum_required(VERSION 3.16)
project(tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)

set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx.proto")

set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnx_generated")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

protobuf_generate_cpp(
    ONNX_SRCS       #first .cc files are fritten to ONNX_SRCS
    ONNX_HDRS       #second .h files are fritten to ONNX_HDRS
    ${ONNX_PROTO}   #which .proto files it will use
    PROTOC_OUT_DIR ${PROTO_GEN_DIR} # to what directive (must be marked as PROTOC_OUT_DIR)
)

#onnx lib
add_library(onnx STATIC ${ONNX_SRCS})

target_include_directories(
    onnx PUBLIC 
        "${PROTO_GEN_DIR}" 
        ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(
    onnx PUBLIC 
        ${Protobuf_LIBRARIES}
)

#tenc_headers lib

file(GLOB TENC_SRC "src/*.cpp")  # Все .cpp файлы
add_library(tenc_lib STATIC ${TENC_SRC})

target_include_directories(tenc_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    PRIVATE
    ${PROTO_GEN_DIR}  # Если нужно для компиляции
)

target_link_libraries(tenc_lib PUBLIC
    onnx  # Если onnx нужен пользователям библиотеки
    PRIVATE
    Threads::Threads
)

file(GLOB SRC_CPP "src/*.cpp")
add_executable(${PROJECT_NAME} ${ONNX_SRCS} ${SRC_CPP})

target_include_directories(
    ${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} 
        ${Protobuf_INCLUDE_DIRS}
        include
)

target_link_libraries(
    ${PROJECT_NAME} PRIVATE
        onnx
        Threads::Threads
)   

enable_testing()
add_subdirectory(tests)
