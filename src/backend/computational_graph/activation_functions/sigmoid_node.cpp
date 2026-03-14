/**
 * @file sigmoid_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "sigmoid_node.h"

#include <utility>

using namespace std;
using namespace cgraph;

vector<shared_ptr<Tensor>> SigmoidNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  constexpr ftype zero = 0.0;
  
  auto res = make_shared<Tensor>(upstreamGrad.getDims(), upstreamGrad.getDevice(), false);

  // s is result from forward pass sigmoid
  auto derivative = [](ftype s){
    return s * (1-s);
  };

  for(tensorSize_t i=0; i<upstreamGrad.getSize(); i++){
    res->set(derivative((*sigmoid)[i] * upstreamGrad[i]), i);
  }

  return {res};
}