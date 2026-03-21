/**
 * @file leaky_relu_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "leaky_relu_node.h"

#include <utility>

using namespace std;
using namespace cgraph;

vector<shared_ptr<Tensor>> LeakyReLuNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  constexpr ftype zero = 0.0;
  
  auto res = make_shared<Tensor>(upstreamGrad.getDims(), upstreamGrad.getDevice(), false);
  
  const auto& parent = parents[0];
  for(tensorSize_t i=0; i<upstreamGrad.getSize(); i++){
    res->set((*parent)[i] > zero ? upstreamGrad[i] : upstreamGrad[i] * eps, i);
  }

  return {res};
}