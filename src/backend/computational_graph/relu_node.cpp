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
  auto res = make_shared<Tensor>(upstreamGrad.getDims().toVector(), upstreamGrad.getDevice(), false);
  for(tensorSize_t i=0; i<upstreamGrad.getSize(); i++){
    (*res)[i] = upstreamGrad[i];
  }
  return {std::move(res)};
}