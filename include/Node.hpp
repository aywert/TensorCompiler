#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <variant>
#include <string>
#include "onnx.pb.h"
#include "Tensor.hpp"

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
  TensorOpType op_type_; //A string identifying the operation.

  std::vector<std::string> node_input_;
  std::vector<std::string> node_output_;

  Tensor output_;  
  //Attributes
  std::unordered_map<std::string, Attribute> attributes_;// meant to have some data structure in order 

  public:
    Node(TensorOpType op_type, std::vector<std::string> node_input, std::vector<std::string> node_output, Tensor output): 
      op_type_(op_type), node_input_(node_input), output_(output) {}

    Node(const onnx::NodeProto& onnx_node);
    void console_dump(size_t order) const;
};

}//tenc