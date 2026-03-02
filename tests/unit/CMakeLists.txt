find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
include(GoogleTest)

file(GLOB TEST_SRC_CPP "src/*.cpp")
add_executable(unit ${TEST_SRC_CPP})

target_include_directories(unit PRIVATE 
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/onnx_generated
)

target_link_libraries(unit
  PRIVATE 
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    tenc_lib 
    onnx
)

gtest_discover_tests(unit
  PROPERTIES LABELS "unit"
)  