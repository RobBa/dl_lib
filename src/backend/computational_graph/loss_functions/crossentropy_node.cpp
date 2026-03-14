/**
 * @file add_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "crossentropy_node.h"

#include "data_modeling/tensor_functions.h"

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > CrossEntropyNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  
  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  for(tensorDim_t i=0; i<static_cast<tensorDim_t>(bSize); i++){
    auto yi = (*yTrue)[i];

    for(tensorDim_t j=0; i<nClasses; i++){
      constexpr ftype eps = 1e-6;
      auto yijHat = std::max(yPred->getItem(i, j), eps);

      auto g = -yi/yijHat;
      res->setItem(g/bSize, i, j);
    }
  }
  
  return {res};
}