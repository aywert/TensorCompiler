#include "Graph.hpp"

namespace tenc {

Graph::Graph(const onnx::GraphProto& graph) {
  //filling up initializers_ - constant tensors that are used by the graph
  for (const onnx::TensorProto& tensor: graph.initializer()) {
    initializers_[tensor.name()] = std::make_unique<Tensor>(tensor);//creating object in heap
  }

  //creating Nodes of the graph using informations about initializers
  for (const onnx::NodeProto& onnx_node : graph.node()) {
    Node node(onnx_node, initializers_);
    vertices_.push_back(node);
  }

  // creating input tensors
  for (const auto& input_proto : graph.input()) {
    if (initializers_.find(input_proto.name()) == initializers_.end()) {
      tensors_[input_proto.name()] = std::make_unique<Tensor>(input_proto);
    }
  }

  link_graph(graph);
}

void Graph::link_graph(const onnx::GraphProto& graph) {

  // for each node from the graph which is not a initializer create a unique_ptr and put it in the tensors_
  for (auto& node : vertices_) {
    for (const std::string& output_name : node.outputs_by_names()) {
      if (initializers_.find(output_name) == initializers_.end() && 
               tensors_.find(output_name) == tensors_.end()) {
        tensors_[output_name] = std::make_unique<Tensor>(output_name);
      } 

      node.push_back_output(tensors_[output_name].get());
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
          node.fill_tensor_input(tensors_[name].get(), i);
        }
      }
    }
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

void Graph::graphviz_dump(std::string filename) {
  std::ofstream out(filename);
  if (!out.is_open()) return;

  out << "digraph ONNX_Graph {\n";
  out << "  rankdir=TB; \n";
  out << "  node [shape=record, style=filled, fontname=\"Arial\"];\n";

  for (auto const& it : initializers_) out << it.second->tensor_label_for_graphviz(true);
  for (auto const& it : tensors_)      out << it.second->tensor_label_for_graphviz(false);


  int node_idx = 0;
  for (const auto& node : vertices_) {
    std::string node_id = "node_" + std::to_string(node_idx++);
    
    out << "  " << node_id << " [label=\"{ Op: " << node.type() 
        << " | Name: " << node.name() << "}\", fillcolor=\"#68dede\"];\n";

    for (size_t i = 0; i < node.get_size_of_input(); ++i) {
      const Tensor* t = node.get_tensor_ptr(i);
      if (t != nullptr) {
        out << "  \"" << t->get_name() << "\" -> " << node_id 
            << " [label=\"in[" << i << "]\"];\n";
        
        //out << "  \"" << t->get_name() << "\" [shape=ellipse, fillcolor=\"#ead7b8\"];\n";
      }
    }

    size_t index = 0;
    
    for (auto* t_out : node.outputs()) {
      if (t_out) {
        out << "  " << node_id << " -> \"" << t_out->get_name() << "\" [label=\"out[" << index++ << "]\"];\n";
      }
    }
  }

  out << "}\n";
  out.close();
  std::cout << "Graph exported to " << filename << std::endl;
}


} //tenc



 