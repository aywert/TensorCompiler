// Graph G is described by a set of vertices (nodes) V 
// and the edges of E which it contains G=(V,E)
#pragma once


#include <iostream>
#include <vector>
#include <fstream>
#include "onnx.pb.h"
#include "Tensor.hpp"
#include "Types.hpp"
#include "Node.hpp"

namespace tenc {

class Graph {
  std::vector<Node> vertices_;

  std::vector<std::string> graph_inputs;
  std::vector<std::string> graph_outputs;

  init_t initializers_; //map of constant tensors
  blob_t tensors_; //map of tensors that are 

  //Initializers

  public:
    Graph() {} 
    Graph(const onnx::GraphProto& graph);
    void link_graph(const onnx::GraphProto& graph);
    void graphviz_dump(std::string filename); 
    void console_dump(void);
};

} //tenc

