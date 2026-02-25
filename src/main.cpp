#include <iostream>
#include <fstream>
#include "onnx.pb.h"
#include "Graph.hpp"


int main(int argc, char* argv[]) {
  onnx::ModelProto onnx_model; // class from generated onnx.pb.cc file     
  std::ifstream input("./examples/conv_layer.onnx", std::ios::in | std::ios::binary); //hard code just for now
  if (!input) {
    std::cerr << "Couldn't open the file" << std::endl;
    return -1;
  }    

  //ParseFromIstream fills up onnx_model
  if (!onnx_model.ParseFromIstream(&input)) {
    std::cerr << "Mistake while parsing" << std::endl;
    return -1;
  }    

  std::cout << "Model succesfully downloaded!" << std::endl;
  
  //graph_builder()
  const onnx::GraphProto& graph = onnx_model.graph();

  tenc::Graph my_graph(graph);
  
  my_graph.console_dump();
  return 0;
}