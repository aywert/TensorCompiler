// tenc_graph_test.cpp
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "onnx.pb.h"
#include "Graph.hpp"   // tenc::Graph
#include "Types.hpp"   // init_t/blob_t typedefs (если нужны в заголовках)
#include "Tensor.hpp"
#include "Node.hpp"

using namespace tenc;
namespace fs = std::filesystem;

// ------------------------------ Helpers ----------------------------------------

static std::string read_file_to_string(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

static onnx::ValueInfoProto make_value_info(const std::string& name,
                                            int elem_type,
                                            std::initializer_list<int64_t> dims) {
    onnx::ValueInfoProto v;
    v.set_name(name);

    auto* tt = v.mutable_type()->mutable_tensor_type();
    tt->set_elem_type(elem_type);

    auto* shape = tt->mutable_shape();
    for (auto d : dims) shape->add_dim()->set_dim_value(d);

    return v;
}

static onnx::TensorProto make_initializer_tensor(const std::string& name,
                                                 int elem_type,
                                                 std::initializer_list<int64_t> dims) {
    onnx::TensorProto t;
    t.set_name(name);
    t.set_data_type(elem_type);
    for (auto d : dims) t.add_dims(d);

    // raw_data можно не заполнять: Graph/Node обычно берут shape+name, а не содержимое.
    return t;
}

static onnx::GraphProto make_minimal_graph_proto() {
    onnx::GraphProto g;
    g.set_name("G");

    // input: X (не initializer)
    *g.add_input() = make_value_info("X", onnx::TensorProto::FLOAT, {1});

    // output: Z
    *g.add_output() = make_value_info("Z", onnx::TensorProto::FLOAT, {1});

    // initializer: Y (константа)
    *g.add_initializer() = make_initializer_tensor("Y", onnx::TensorProto::FLOAT, {1});

    // node: Add_0: Z = X + Y
    auto* n = g.add_node();
    n->set_name("Add_0");
    n->set_op_type("Add");
    n->add_input("X");
    n->add_input("Y");
    n->add_output("Z");

    return g;
}

// ----------------------------- Constructors ------------------------------------

TEST(TencGraph, DefaultConstructorDoesNotCrash) {
    Graph gr;
    // просто smoke-test
    EXPECT_NO_THROW(gr.console_dump());
}

TEST(TencGraph, ConstructFromOnnxGraphProtoDoesNotCrash) {
    auto proto = make_minimal_graph_proto();
    EXPECT_NO_THROW({
        Graph gr(proto);
        gr.console_dump();
    });
}

// ------------------------------ link_graph ------------------------------------

TEST(TencGraph, LinkGraphBuildsInternalStateAndDoesNotCrash) {
    Graph gr;
    auto proto = make_minimal_graph_proto();

    EXPECT_NO_THROW(gr.link_graph(proto));
    EXPECT_NO_THROW(gr.console_dump());
}