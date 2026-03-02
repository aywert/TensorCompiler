// tenc_node_test.cpp
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnx.pb.h"

// ВАЖНО: подключай тот header, где объявлен tenc::Node (у тебя он называется Node.hpp)
#include "Node.hpp"
#include "Tensor.hpp"
#include "Types.hpp"

using namespace tenc;

// ------------------------------ Helpers ----------------------------------------

static onnx::AttributeProto make_attr_int(const std::string& name, int64_t v) {
    onnx::AttributeProto a;
    a.set_name(name);
    a.set_type(onnx::AttributeProto::INT);
    a.set_i(v);
    return a;
}

static onnx::AttributeProto make_attr_float(const std::string& name, float v) {
    onnx::AttributeProto a;
    a.set_name(name);
    a.set_type(onnx::AttributeProto::FLOAT);
    a.set_f(v);
    return a;
}

static onnx::AttributeProto make_attr_string(const std::string& name, const std::string& v) {
    onnx::AttributeProto a;
    a.set_name(name);
    a.set_type(onnx::AttributeProto::STRING);
    a.set_s(v);
    return a;
}

static onnx::AttributeProto make_attr_ints(const std::string& name, std::initializer_list<int64_t> vs) {
    onnx::AttributeProto a;
    a.set_name(name);
    a.set_type(onnx::AttributeProto::INTS);
    for (auto x : vs) a.add_ints(x);
    return a;
}

static onnx::AttributeProto make_attr_floats(const std::string& name, std::initializer_list<float> vs) {
    onnx::AttributeProto a;
    a.set_name(name);
    a.set_type(onnx::AttributeProto::FLOATS);
    for (auto x : vs) a.add_floats(x);
    return a;
}

// ----------------------------- Ctor (manual) ----------------------------------
// Node(std::string name, TensorOpType op_type, std::vector<std::string> node_input,
//      std::vector<std::string> node_output, Tensor output)
// У тебя в initializer list заполняются только name_/op_type_/input_by_names_.

TEST(TencNode, ManualConstructorStoresNameTypeAndInputsOnly) {
    Tensor dummy; // предполагаем, что Tensor имеет default ctor
    std::vector<std::string> in  = {"x", "y"};
    std::vector<std::string> out = {"z"}; // в текущей реализации игнорируется

    Node n("MyNode", "Add", in, out, dummy);

    EXPECT_EQ(n.name(), "MyNode");
    EXPECT_EQ(n.type(), "Add");

    ASSERT_EQ(n.get_size_of_input(), 2u);
    EXPECT_EQ(n.get_name_of_input(0), "x");
    EXPECT_EQ(n.get_name_of_input(1), "y");

    // В текущем коде output_by_names_ не инициализируется этим ctor’ом
    EXPECT_EQ(n.get_size_of_output(), 0u);
    EXPECT_TRUE(n.outputs_by_names().empty());

    // input_/output_ тоже не заполняются этим ctor’ом
    EXPECT_TRUE(n.inputs().empty());
    EXPECT_TRUE(n.outputs().empty());
}

// --------------------------- search_in_initializer ----------------------------
// Проверяем опосредованно через onnx-ctor: input_ заполняется либо ptr, либо nullptr.

TEST(TencNode, OnnxConstructorFillsInputsWithInitializerPointersOrNull) {
    init_t initializers;

    // Кладём один тензор в initializers
    initializers["W"] = std::make_unique<Tensor>();
    const Tensor* wptr = initializers["W"].get();

    onnx::NodeProto proto;
    proto.set_name("Conv_0");
    proto.set_op_type("Conv");

    // inputs: "X" (нет в init) и "W" (есть)
    proto.add_input("X");
    proto.add_input("W");

    // outputs
    proto.add_output("Y");

    Node n(proto, initializers);

    EXPECT_EQ(n.name(), "Conv_0");
    EXPECT_EQ(n.type(), "Conv");

    ASSERT_EQ(n.get_size_of_input(), 2u);
    EXPECT_EQ(n.get_name_of_input(0), "X");
    EXPECT_EQ(n.get_name_of_input(1), "W");

    ASSERT_EQ(n.inputs().size(), 2u);
    EXPECT_EQ(n.get_tensor_ptr(0), nullptr);
    EXPECT_EQ(n.get_tensor_ptr(1), wptr);

    ASSERT_EQ(n.get_size_of_output(), 1u);
    EXPECT_EQ(n.get_name_of_output(0), "Y");
}

// ------------------------------- Attributes -----------------------------------
// Прямого getter для attributes_ нет, поэтому тестируем через console_dump().

TEST(TencNode, OnnxConstructorParsesAllSupportedAttributeTypes) {
    init_t initializers;

    onnx::NodeProto proto;
    proto.set_name("Gemm_0");
    proto.set_op_type("Gemm");
    proto.add_input("A");
    proto.add_input("B");
    proto.add_output("C");

    *proto.add_attribute() = make_attr_int("transA", 1);
    *proto.add_attribute() = make_attr_float("alpha", 0.25f);
    *proto.add_attribute() = make_attr_string("auto_pad", "SAME_UPPER");
    *proto.add_attribute() = make_attr_ints("kernel_shape", {3, 3});
    *proto.add_attribute() = make_attr_floats("scales", {0.5f, 1.5f});

    Node n(proto, initializers);

    testing::internal::CaptureStdout();
    n.console_dump(7);
    std::string out = testing::internal::GetCapturedStdout();

    // базовые поля
    EXPECT_NE(out.find("Node: 7"), std::string::npos);
    EXPECT_NE(out.find("Name: Gemm_0"), std::string::npos);
    EXPECT_NE(out.find("op_type: Gemm"), std::string::npos);

    // inputs / outputs
    EXPECT_NE(out.find("inputs:"), std::string::npos);
    EXPECT_NE(out.find("A"), std::string::npos);
    EXPECT_NE(out.find("B"), std::string::npos);

    EXPECT_NE(out.find("outputs:"), std::string::npos);
    EXPECT_NE(out.find("C"), std::string::npos);

    // атрибуты (проверяем по подстрокам, т.к. порядок в unordered_map не гарантирован)
    EXPECT_NE(out.find("Attributes:"), std::string::npos);

    EXPECT_NE(out.find("name: transA"), std::string::npos);
    EXPECT_NE(out.find("value: 1"), std::string::npos);

    EXPECT_NE(out.find("name: alpha"), std::string::npos);
    // float печатается через std::cout (может быть "0.25" или "0.25..."), поэтому ищем "0.25"
    EXPECT_NE(out.find("0.25"), std::string::npos);

    EXPECT_NE(out.find("name: auto_pad"), std::string::npos);
    EXPECT_NE(out.find("SAME_UPPER"), std::string::npos);

    EXPECT_NE(out.find("name: kernel_shape"), std::string::npos);
    EXPECT_NE(out.find("[ 3 3 ]"), std::string::npos);

    EXPECT_NE(out.find("name: scales"), std::string::npos);
    EXPECT_NE(out.find("[ 0.5 1.5 ]"), std::string::npos);
}

TEST(TencNode, OnnxConstructorIgnoresUnsupportedAttributeTypes) {
    init_t initializers;

    onnx::NodeProto proto;
    proto.set_name("Weird_0");
    proto.set_op_type("CustomOp");
    proto.add_input("X");
    proto.add_output("Y");

    // поддерживаемый
    *proto.add_attribute() = make_attr_int("axis", 2);

    // неподдерживаемый (у тебя нет default: case, но и нет case для TENSOR)
    {
        onnx::AttributeProto a;
        a.set_name("ignored_tensor");
        a.set_type(onnx::AttributeProto::TENSOR);
        *proto.add_attribute() = a;
    }

    Node n(proto, initializers);

    testing::internal::CaptureStdout();
    n.console_dump(1);
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("name: axis"), std::string::npos);
    EXPECT_EQ(out.find("ignored_tensor"), std::string::npos);
}

// -------------------------- fill_tensor_input/output ---------------------------

TEST(TencNode, FillTensorInputReplacesNullAtIndex) {
    init_t initializers;

    onnx::NodeProto proto;
    proto.set_name("N");
    proto.set_op_type("Add");
    proto.add_input("X"); // будет nullptr
    proto.add_input("Y"); // будет nullptr
    proto.add_output("Z");

    Node n(proto, initializers);

    ASSERT_EQ(n.inputs().size(), 2u);
    EXPECT_EQ(n.get_tensor_ptr(0), nullptr);

    Tensor x;
    n.fill_tensor_input(&x, 0);

    EXPECT_EQ(n.get_tensor_ptr(0), &x);
    EXPECT_EQ(n.get_tensor_ptr(1), nullptr);
}

TEST(TencNode, FillTensorOutputAndPushBackOutputWork) {
    init_t initializers;

    onnx::NodeProto proto;
    proto.set_name("N");
    proto.set_op_type("Relu");
    proto.add_input("X");
    proto.add_output("Y");

    Node n(proto, initializers);

    // output_ в onnx-ctor не наполняется, но мы можем пушить вручную
    EXPECT_TRUE(n.outputs().empty());

    Tensor y;
    n.push_back_output(&y);

    ASSERT_EQ(n.outputs().size(), 1u);
    EXPECT_EQ(n.outputs()[0], &y);

    Tensor y2;
    n.fill_tensor_output(&y2, 0);
    ASSERT_EQ(n.outputs().size(), 1u);
    EXPECT_EQ(n.outputs()[0], &y2);
}

// ------------------------------- Name getters ---------------------------------

TEST(TencNode, InputsOutputsNameVectorsReturnCopiesWithCorrectContent) {
    init_t initializers;

    onnx::NodeProto proto;
    proto.set_name("N");
    proto.set_op_type("Mul");
    proto.add_input("A");
    proto.add_input("B");
    proto.add_output("C");
    proto.add_output("D");

    Node n(proto, initializers);

    auto ins = n.inputs_by_names();
    auto outs = n.outputs_by_names();

    ASSERT_EQ(ins.size(), 2u);
    EXPECT_EQ(ins[0], "A");
    EXPECT_EQ(ins[1], "B");

    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0], "C");
    EXPECT_EQ(outs[1], "D");
}