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

size_t Tensor::get_type_size(DataType type) {
  switch (type) {
    case DataType::FLOAT:  return sizeof(float);
    case DataType::INT32:  return sizeof(int32_t);
    case DataType::INT64:  return sizeof(int64_t);
    case DataType::DOUBLE: return sizeof(double);
    case DataType::BOOL:   return sizeof(bool);
    default:               return 0;
  }
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

} //tenc