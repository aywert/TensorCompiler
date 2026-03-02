
// tenc_tensor_test.cpp
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "onnx.pb.h"
#include "Tensor.hpp"

using namespace tenc;

// ------------------------------ Helpers ----------------------------------------

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static void expect_type_name_contains(const std::string& type_name, const std::string& needle_lower) {
    const std::string got = to_lower(type_name);
    EXPECT_NE(got.find(needle_lower), std::string::npos)
        << "Expected type name to contain '" << needle_lower << "', got: '" << type_name << "'";
}

static void expect_shape_string_contains_dims(const std::string& shape_str,
                                             const std::vector<int64_t>& dims) {
    for (auto d : dims) {
        const std::string token = std::to_string(d);
        EXPECT_NE(shape_str.find(token), std::string::npos)
            << "Expected shape string to contain dim '" << token << "', got: '" << shape_str << "'";
    }
}

// ----------------------------- Constructors ------------------------------------

TEST(TencTensor, DefaultConstructorInitializesDefaultNameAndUndefinedType) {
    Tensor t;

    EXPECT_EQ(t.get_name(), "default_tensor");
    EXPECT_EQ(t.get_data_type(), DataType::UNDEFINED);
    EXPECT_TRUE(t.get_shape().empty());

    // get_type_name() должен корректно отразить UNDEFINED (проверяем мягко)
    expect_type_name_contains(t.get_type_name(), "undef");
}

TEST(TencTensor, NameConstructorStoresNameAndKeepsUndefinedType) {
    Tensor t("X");

    EXPECT_EQ(t.get_name(), "X");
    EXPECT_EQ(t.get_data_type(), DataType::UNDEFINED);
    EXPECT_TRUE(t.get_shape().empty());

    expect_type_name_contains(t.get_type_name(), "undef");
}

TEST(TencTensor, ParameterizedConstructorStoresNameTypeShapeData) {
    std::string name = "W";
    std::vector<int64_t> shape = {2, 3, 4};
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};

    Tensor t(name, DataType::INT64, shape, data);

    EXPECT_EQ(t.get_name(), "W");
    EXPECT_EQ(t.get_data_type(), DataType::INT64);

    const auto& got_shape = t.get_shape();
    ASSERT_EQ(got_shape.size(), 3u);
    EXPECT_EQ(got_shape[0], 2);
    EXPECT_EQ(got_shape[1], 3);
    EXPECT_EQ(got_shape[2], 4);

    expect_type_name_contains(t.get_type_name(), "int64");

    const std::string shape_str = t.get_shape_string();
    expect_shape_string_contains_dims(shape_str, shape);
}

// -------------------------- ONNX TensorProto constructor -----------------------

TEST(TencTensor, OnnxTensorProtoConstructorReadsNameTypeAndShape) {
    onnx::TensorProto proto;
    proto.set_name("T");
    proto.set_data_type(onnx::TensorProto::FLOAT);  // ONNX enum value
    proto.add_dims(1);
    proto.add_dims(5);

    // raw_data можно не заполнять — публичных геттеров для data_ всё равно нет
    Tensor t(proto);

    EXPECT_EQ(t.get_name(), "T");
    EXPECT_EQ(t.get_data_type(), DataType::FLOAT);

    const auto& s = t.get_shape();
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[1], 5);

    expect_type_name_contains(t.get_type_name(), "float");

    const std::string shape_str = t.get_shape_string();
    expect_shape_string_contains_dims(shape_str, {1, 5});
}

// ------------------------ ONNX ValueInfoProto constructor ----------------------

TEST(TencTensor, OnnxValueInfoProtoConstructorReadsNameTypeAndShape) {
    onnx::ValueInfoProto info;
    info.set_name("input_0");

    auto* tt = info.mutable_type()->mutable_tensor_type();
    tt->set_elem_type(onnx::TensorProto::INT32);

    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(7);

    Tensor t(info);

    EXPECT_EQ(t.get_name(), "input_0");
    EXPECT_EQ(t.get_data_type(), DataType::INT32);

    const auto& s = t.get_shape();
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0], 3);
    EXPECT_EQ(s[1], 7);

    expect_type_name_contains(t.get_type_name(), "int32");

    const std::string shape_str = t.get_shape_string();
    expect_shape_string_contains_dims(shape_str, {3, 7});
}

// ------------------------------ console_dump ----------------------------------

TEST(TencTensor, ConsoleDumpPrintsSomethingUseful) {
    Tensor t("A");
    testing::internal::CaptureStdout();
    t.console_dump();
    std::string out = testing::internal::GetCapturedStdout();

    // Очень мягкие проверки, чтобы не зависеть от точного формата
    EXPECT_NE(out.find("A"), std::string::npos);
}

// ------------------------- tensor_label_for_graphviz ---------------------------

TEST(TencTensor, TensorLabelForGraphvizContainsName) {
    Tensor t("X");

    const std::string label_init = t.tensor_label_for_graphviz(true);
    const std::string label_non  = t.tensor_label_for_graphviz(false);

    EXPECT_NE(label_init.find("X"), std::string::npos);
    EXPECT_NE(label_non.find("X"), std::string::npos);
}

TEST(TencTensor, TensorLabelForGraphvizContainsShapeWhenPresent) {
    Tensor t("W", DataType::FLOAT, {2, 3}, /*data*/{});

    const std::string label = t.tensor_label_for_graphviz(true);

    EXPECT_NE(label.find("W"), std::string::npos);
    // мягко: просто убеждаемся, что числа размерностей где-то есть
    EXPECT_NE(label.find("2"), std::string::npos);
    EXPECT_NE(label.find("3"), std::string::npos);
}