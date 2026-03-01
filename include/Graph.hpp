// Graph G is described by a set of vertices (nodes) V 
// and the edges of E which it contains G=(V,E)
#pragma once

#include <iostream>
#include <vector>
#include "onnx.pb.h"
#include "Node.hpp"
#include "Edge.hpp"

namespace tenc {

class Graph {
  std::vector<Node> vertices_;
  
  std::vector<std::string> graph_inputs;
  std::vector<std::string> graph_outputs;

  std::map<std::string, std::unique_ptr<Tensor>> initializers_; //constant tensors of the model

  //Initializers

  public:
    Graph() {} 
    Graph(const onnx::GraphProto& graph);
    void console_dump(void);
};

} //tenc

