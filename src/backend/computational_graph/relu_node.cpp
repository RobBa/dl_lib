/**
 * @file relu_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-15
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "relu_node.h"

#include <utility>

using namespace std;
using namespace graph;

vector<shared_ptr<Tensor>> ReLuNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  constexpr ftype zero = 0.0;
  
  auto res = make_shared<Tensor>(upstreamGrad.getDims().toVector(), upstreamGrad.getDevice(), false);
  for(tensorSize_t i=0; i<upstreamGrad.getSize(); i++){
    auto v = upstreamGrad.getItem(i);
    res->setItem(v > zero ? v : zero, i);
  }
  return {std::move(res)};
}