#include <iostream>
#include <fstream>
#include "onnx.pb.h"
#include "Graph.hpp"


int main(int argc, char* argv[]) {
  if (argc != 2) return 0;
  
  onnx::ModelProto onnx_model; // class from generated onnx.pb.cc file     
  std::ifstream input(argv[1], std::ios::in | std::ios::binary); //hard code just for now
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
  
  const onnx::GraphProto& graph = onnx_model.graph();
  tenc::Graph my_graph(graph);
  
  //my_graph.console_dump();
  my_graph.graphviz_dump("./graphviz/graph.dot");

  return 0;
}