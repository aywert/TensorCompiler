#include "Graph.hpp"

namespace tenc {

Graph::Graph(const onnx::GraphProto& graph) {
  for (const onnx::TensorProto& tensor: graph.initializer()) {
    initializers_[tensor.name()] = std::make_unique<Tensor>(tensor);//creating object in heap
  }

  for (const onnx::NodeProto& onnx_node : graph.node()) {
    Node node(onnx_node, initializers_);
    vertices_.push_back(node);
  }
}

void Graph::console_dump() {
  size_t num = 0;
  for (const Node& node : vertices_) {
    node.console_dump(num); num++;
    std::cout << "=============\n";
  }

  for (const auto& tensor : initializers_) {
    tensor.second->console_dump();
    std::cout << "=============\n";
  }
}

void link_graph() {
  
}


}