#pragma once

#include "Tensor.hpp"
#include "Node.hpp"

namespace tenc {

class Edge {
  Tensor data_;

  Node* producer_ = nullptr;
  std::vector<Node*> consumers_;

  public: 
    Edge(Tensor data,   Node* producer,    std::vector<Node*> consumers): 
                data_(data),  producer_(producer), consumers_(consumers) {}
};

} //tenc