#include "Graph.hpp"

namespace tenc {

Graph::Graph(const onnx::GraphProto& graph) {
  for (const onnx::NodeProto& onnx_node : graph.node()) {
    Node node(onnx_node);
    vertices_.push_back(node);
  }
}

void Graph::console_dump() {
  size_t num = 0;
  for (const Node& node : vertices_) {
    node.console_dump(num); num++;
    std::cout << "=============\n";
  }
}

  

}