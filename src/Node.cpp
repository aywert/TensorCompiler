#include "Node.hpp"

namespace tenc {
Node::Node(const onnx::NodeProto& onnx_node) {
  op_type_ = onnx_node.op_type();

  for (const auto& input_name : onnx_node.input()) {
    node_input_.push_back(input_name);
  }
    
  for (const auto& output_name : onnx_node.output()) {
    node_output_.push_back(output_name);
  }

  //default constructor for Tensor output_;

  for (const auto& attr : onnx_node.attribute()) {
    std::string name = attr.name();
    
    switch(attr.type()) {
      case onnx::AttributeProto::INT: {
        attributes_[name] = attr.i();
        break;
      }
          
      case onnx::AttributeProto::FLOAT: {
        attributes_[name] = attr.f();
        break;
      }
          
      case onnx::AttributeProto::STRING: {
        attributes_[name] = attr.s();
        break;
      }
          
      case onnx::AttributeProto::INTS: {
        std::vector<int64_t> v_int;
        for (int i = 0; i < attr.ints_size(); ++i) {
          v_int.push_back(attr.ints(i));
        }
        attributes_[name] = v_int;
        break;
      }
          
      case onnx::AttributeProto::FLOATS: { 
        std::vector<int64_t> v_float;
        for (int i = 0; i < attr.floats_size(); ++i) {
            v_float.push_back(attr.floats(i));
        }
        attributes_[name] = v_float;
        break;
      }
      
    }
  } //for
} //class Node

void Node::console_dump(size_t order) const {
  std::cout << "Node: " << order << "\n";
  std::cout << "op_type: " << op_type_ << "\n";

  std::cout << "inputs: ";
  for (const auto& input_name : node_input_) {
    std::cout << input_name << " ";
  }

  std::cout << "\n";

  std::cout << "outputs: ";
  for (const auto& output_name : node_output_) {
    std::cout << output_name << " ";
  }

  std::cout << "\n";
}


} //tenc