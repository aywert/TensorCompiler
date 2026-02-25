// Graph G is described by a set of vertices (nodes) V 
// and the edges of E which it contains G=(V,E)
#pragma once

#include <iostream>
#include <vector>
#include "Node.hpp"
#include "Edge.hpp"

namespace tenc {

class Graph {
  std::vector<Node> vertices_;
  std::vector<Edge> edges_;

  public:
    Graph() {} 
};

} //tenc