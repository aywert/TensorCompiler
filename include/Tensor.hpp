#pragma once

#include <iostream>
#include <vector>
#include <any>

namespace tenc {

enum class DataType {
  Undef_t,
  Double_t,
  Int_t,
  Bool_t,
  String_t
};

class Tensor final {
  std::string name_;
  DataType type_ = DataType::Undef_t;
  std::vector<int64_t> shape_; // sizes of tensor
  std::any data_;

  public: 
    Tensor(std::string name, DataType type, std::vector<int64_t> shape, std::any data): 
                       name_(name),   type_(type),        shape_(shape),   data_(data) {}

    const std::string& name()           const { return name_;  }
    DataType data_type()                const { return type_;  }
    const std::vector<int64_t>& shape() const { return shape_; }
};

} //tenc