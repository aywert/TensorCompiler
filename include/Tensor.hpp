#pragma once

#include <iostream>
#include <vector>
#include <any>
#include "onnx.pb.h"

namespace tenc {

enum DataType {
  UNDEFINED,
  INT32,
  INT64,
  FLOAT,
  DOUBLE,
  BOOL,
  STRING
};  

class Tensor final {
  std::string name_;
  DataType type_ = DataType::UNDEFINED;

  std::vector<uint8_t> data_; // raw data contains in 1 byte vector(unsigned char)
  std::vector<int64_t> shape_; // sizes of tensor
  std::vector<int64_t> strides_;

  static DataType convert_onnx_data_type(int32_t onnx_data_type);
  void copy_data_from_onnx(const onnx::TensorProto& tensor);
  size_t get_type_size() const; 
  void calculate_strides();

  public: 
    Tensor(std::string name, DataType type, std::vector<int64_t> shape, std::vector<uint8_t> data): 
                       name_(name),   type_(type),        shape_(shape),   data_(data) {}
    Tensor() {name_ = "default_tensor"; type_ = DataType::UNDEFINED;};
    Tensor(const std::string& name): name_(name) {}
    Tensor(const onnx::TensorProto& tensor);
    Tensor(const onnx::ValueInfoProto& info);

    void console_dump() const;

    const std::vector<int64_t>& get_shape() const { return shape_; }
    const std::string& get_name()           const { return name_;  }
    const std::string get_type_name()       const; 
    DataType get_data_type()                const { return type_;  }
    std::string get_shape_string()          const;
    std::string tensor_label_for_graphviz (bool is_init) const;
};

} //tenc