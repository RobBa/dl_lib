/**
 * @file softmax_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-15
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "softmax_node.h"

#include "data_modeling/tensor_functions.h"

#include <iostream>

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > SoftmaxNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  constexpr ftype eps = 1e-9;
  
  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  ftype bSize = yPred->getDims()[0];
  assert(bSize>0);
  for(tensorDim_t i=0; i<yPred->getDims()[0]; i++){
    for(tensorDim_t j=0; j<yPred->getDims()[1]; j++){
      ftype g = 0;

      if(i!=j){
        g = -softmax->get(i) * softmax->get(j);
      }
      else{
        g = softmax->get(i) * (1-softmax->get(j));
      }

      res->set(upstreamGrad[i] * g / bSize, i, j);
    }
  }
  
  return {res};
}