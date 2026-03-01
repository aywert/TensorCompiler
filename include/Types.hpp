#pragma once
#include <map>
#include <string>
#include <iostream>
#include "Tensor.hpp"

namespace tenc {
  using init_t = std::map<std::string, std::unique_ptr<const Tensor>>;
  using blob_t = std::map<std::string, std::unique_ptr<Tensor>>; //iconic name for intermidiate representation
}