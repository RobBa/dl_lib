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

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > BceNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  ftype bSize = yPred->getDims()[0];
  for(tensorSize_t i=0; i<yPred->getDims()[0]; i++){
    auto yi = (*yTrue)[i];
    auto yiHat = (*yPred)[i];

    auto g = -yi/std::max(yiHat, epsBce) + (1-yi)/std::max(1-yiHat, epsBce);
    res->set(g/bSize, i);
  }
  
  return {res};
}