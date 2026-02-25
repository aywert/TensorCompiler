#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include "Tensor.hpp"

namespace tenc {

enum class TensorOpType{
  Add, 
  Mul, 
  Conv, 
  Relu, 
  MatMul, 
  Gemm,
};


class Node {
  TensorOpType op_type_; //A string identifying the operation.

  std::vector<std::string> node_input_;
  std::vector<std::string> node_output_;

  Tensor output_;  
  //Attributes
  //std::unordered_map<std::string, Attribute> attributes_; meant to have some data structure in order 

  public:
    Node(TensorOpType op_type, std::vector<std::string> node_input, std::vector<std::string> node_output, Tensor output): 
      op_type_(op_type), node_input_(node_input), output_(output) {}
};

}//tenc