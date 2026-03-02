#include "Tensor.hpp"
namespace tenc {

Tensor::Tensor(const onnx::TensorProto& tensor) {
  name_ = tensor.name();
  type_ = convert_onnx_data_type(tensor.data_type());

  for ( int dim = 0; dim < tensor.dims_size(); dim++) {
    shape_.push_back(tensor.dims(dim));
  }

  copy_data_from_onnx(tensor); 
  calculate_strides(); 
}

Tensor::Tensor(const onnx::ValueInfoProto& info) {
  name_ = info.name();
  const auto& type_proto = info.type().tensor_type();
  type_ = convert_onnx_data_type(type_proto.elem_type());

  for (int i = 0; i < type_proto.shape().dim_size(); ++i) {
      shape_.push_back(type_proto.shape().dim(i).dim_value());
  }

  calculate_strides();
}

DataType Tensor::convert_onnx_data_type(int32_t onnx_data_type) {
  switch (onnx_data_type) {
    case onnx::TensorProto::INT32:   return DataType::INT32;
    case onnx::TensorProto::INT64:   return DataType::INT64;
    case onnx::TensorProto::FLOAT:   return DataType::FLOAT;
    case onnx::TensorProto::DOUBLE:  return DataType::DOUBLE;
    case onnx::TensorProto::BOOL:    return DataType::BOOL;
    case onnx::TensorProto::STRING:  return DataType::STRING;
    default: return DataType::UNDEFINED;
  }
}

size_t Tensor::get_type_size() const {
  switch (type_) {
    case DataType::FLOAT:  return sizeof(float);
    case DataType::INT32:  return sizeof(int32_t);
    case DataType::INT64:  return sizeof(int64_t);
    case DataType::DOUBLE: return sizeof(double);
    case DataType::BOOL:   return sizeof(bool);
    default:               return 0;
  }
}

const std::string Tensor::get_type_name() const { 
  switch (type_) {
    case DataType::FLOAT:  return "float";
    case DataType::INT32:  return "int32";
    case DataType::INT64:  return "int64";
    case DataType::DOUBLE: return "double";
    case DataType::BOOL:   return "bool";
    default:               return "undef";
  }
}

std::string Tensor::get_shape_string() const {
  std::string result;
  for (int64_t i = 0; i < shape_.size(); ++i) {
    result+= std::to_string(shape_[i]);
    if (i != shape_.size()-1) result += "x";
  }
  return result;
}

void Tensor::copy_data_from_onnx(const onnx::TensorProto& tensor) {
  const std::string& raw = tensor.raw_data();

  if(!raw.empty()) {
    data_.resize(raw.size());
    std::memcpy(data_.data(), raw.data(), raw.size());
  }
}

void Tensor::calculate_strides() { //usage of strides: call data_[i][j][k] -> data[offset]; offset = i * strides_[0] + j * strides_[1] + k * strides_[2];
  strides_.resize(shape_.size());
  if (shape_.empty()) return;

  int64_t current_stride = 1;
  for (int i = shape_.size() - 1; i >= 0; --i) {
    strides_[i] = current_stride;
    current_stride *= shape_[i];
  }
}

void Tensor::console_dump() const {
  std::cout << "Tensor: " << name_ << "\n";
  std::cout << "Size: " << get_type_size() << "\n";
  std::cout << "Shape [ ";
  for (const auto& num: shape_) std::cout << num << " ";
  std::cout << "]\n";
  std::cout << "Strides [ ";
  for (const auto& num: strides_) std::cout << num << " ";
  std::cout << "]\n";

  std::cout << std::flush;
}

std::string Tensor::tensor_label_for_graphviz (bool is_init) const {
      std::string color = is_init ? "#ead7b8" : "#ffffff"; // Веса - бежевые, данные - белые
      std::string label = "{ " + get_name() + " | " 
                        + get_type_name() + " | " 
                        + get_shape_string() + " }";
      
      std::string result =  "  \"" + get_name() + "\" [shape=record, style=filled, fillcolor=\"" 
          + color + "\", label=\"" + label + "\"];\n";
      return result;
    };

} //tenc