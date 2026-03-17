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

  ftype bSize = yPred->getDims()[0];
  for(tensorDim_t i=0; i<yPred->getDims()[0]; i++){
    for(tensorDim_t j=0; j<yPred->getDims()[1]; j++){
      auto yij = yTrue->get(i, j);
      auto yijHat = yPred->get(i, j);

      auto g = -yij/std::max(yijHat, epsCrossentropy);
      res->set(g/bSize, i, j);
    }
  }
  
  return {res};
}