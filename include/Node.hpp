#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <variant>
#include <string>
#include "onnx.pb.h"
#include "Tensor.hpp"
#include "Types.hpp"


////////////////////////////////////////////////////////////////////////////////
//Types of attributes                                                         //
//                                                                            //
// CONV                                         MATMUL RELU - no attributes   //                          
// Attribute      Value                         ADD & MUL                     //
// kernel_shape	  std::vector<int64_t>,	        axis         int64_t          //
// strides	      std::vector<int64_t>,	        broadcast    int64_t          //
// dilations	    std::vector<int64_t>,	                                      //
// pads	          std::vector<int64_t>,	                                      //
// auto_pad	      std::string	                                                //
// group	        int64_t	                                                    //
//                                                                            //
// GEMM                                                                       //
// alpha	float	                                                              //
// beta	  float	                                                              //
// transA	int64_t                                                             //
// transB	int64_t                                                             //
//                                                                            //
// MATMUL RELU - no attributes                                                //
// ADD & MUL                                                                  //
// axis         int64_t                                                       //
// broadcast    int64_t                                                       //
////////////////////////////////////////////////////////////////////////////////

namespace tenc {

using TensorOpType = std::string;
using Attribute = std::variant<
  int64_t,                    
  float,                                   
  std::vector<int64_t>,  
  std::vector<float>,   
  std::string,       
  std::vector<std::string>    
>;

using PairStrAttr = std::pair<const std::string, Attribute>;

class Node {
  std::string name_;
  TensorOpType op_type_; //A string identifying the operation.

  std::vector<std::string> node_input_;
  std::vector<std::string> node_output_;
  std::vector<const Tensor*> input_;
  std::vector<Tensor*> output_;
  //Tensor output_;

  //Attributes
  std::unordered_map<std::string, Attribute> attributes_;// meant to have some data structure in order 

  const Tensor* search_in_initializer(const std::string& name, const init_t& initializers) const;

  public:
    Node(std::string name, TensorOpType op_type, std::vector<std::string> node_input, std::vector<std::string> node_output, Tensor output): 
    name_(name), op_type_(op_type), node_input_(node_input) {}

    Node(const onnx::NodeProto& onnx_node, init_t& initializers);
    void console_dump(size_t order) const;

    const Tensor* get_tensor_ptr(size_t index) const      { return input_[index]; }
    void fill_tensor_input(const Tensor* tensor, size_t index) { input_[index]  = tensor; }
    void fill_tensor_output(Tensor* tensor, size_t index)      { output_[index] = tensor; }

    std::string name() const {return name_;}
    std::string type() const {return op_type_;}
    size_t get_size_of_input() const  {return node_input_.size(); }
    size_t get_size_of_output() const {return node_output_.size();}

    std::string get_name_of_input(size_t index)  const {return node_input_[index];}
    std::string get_name_of_output(size_t index) const {return node_output_[index];}

    void add_output(Tensor* tensor) {output_.push_back(tensor);}
    std::vector<std::string>    str_inputs()  const {return node_input_;}
    std::vector<std::string>    str_outputs() const {return node_output_;}
    std::vector<const Tensor*>  inputs()      const {return input_;}
    std::vector<Tensor*>        outputs()     const {return output_;}
};

} //tenc