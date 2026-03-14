/**
 * @file bce_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "bce_node.h"

#include "data_modeling/tensor_functions.h"

#include <iostream>

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > BceNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  for(tensorSize_t i=0; i<static_cast<tensorSize_t>(bSize); i++){
    auto yi = (*yTrue)[i];
    auto yiHat = (*yPred)[i];

    constexpr ftype eps = 1e-6;
    auto g = -yi/std::max(yiHat, eps) + (1-yi)/std::max(1-yiHat, eps);
    res->setItem(g/bSize, i);
  }
  
  return {res};
}