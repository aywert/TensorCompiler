#include "Graph.hpp"

namespace tenc {

Graph::Graph(const onnx::GraphProto& graph) {
  //filling up initializers_ - constant tensors that are used by the graph
  for (const onnx::TensorProto& tensor: graph.initializer()) {
    initializers_[tensor.name()] = std::make_unique<Tensor>(tensor);//creating object in heap
  }

  //creating Nodes of the graph
  for (const onnx::NodeProto& onnx_node : graph.node()) {
    Node node(onnx_node, initializers_);
    vertices_.push_back(node);
  }

  // creating input tensors
  for (const auto& input_proto : graph.input()) {
    if (initializers_.find(input_proto.name()) == initializers_.end()) {
      tensors_[input_proto.name()] = std::make_unique<Tensor>(input_proto.name());
    }
  }

  link_graph(graph);
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

void Graph::link_graph(const onnx::GraphProto& graph) {

  // for each node from the graph which is not a initializer create a unique_ptr and put it in the tensors_
  for (const auto& node_proto : graph.node()) {
    for (const std::string& output_name : node_proto.output()) {
      if (initializers_.find(output_name) == initializers_.end()) {
        tensors_[output_name] = std::make_unique<Tensor>(output_name);
      }
    }
  }

  // for each vertice find appropriate tensor ptr in tensors_
  for (auto& node : vertices_) {
    size_t size = node.get_size_of_input();
    for (size_t i = 0; i < size; ++i) {
      //marked nullptr are the fields for the blob_t tensors 
      if (node.get_tensor_ptr(i) == nullptr) { 
        std::string name = node.get_name_of_input(i);

        if (tensors_.count(name)) {
          node.fill_tensor(tensors_[name].get(), i);
        }
      }
    }
  }
}

}