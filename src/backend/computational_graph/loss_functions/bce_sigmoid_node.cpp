/**
 * @file bce_sigmoid_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-17
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "bce_sigmoid_node.h"

#include "data_modeling/tensor_functions.h"

#include <cmath>

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > BceSigmoidNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  
  auto sigmoid = [](ftype x){
    constexpr ftype one = 1.0;
    if(x>=0){
      return one / (one + exp(-x));
    }
    auto e = exp(x);
    return e / (one + e);
  };

  const auto& logits = parents[0];
  auto res = make_shared<Tensor>(logits->createEmptyCopy());

  ftype bSize = logits->getDims()[0];
  for(tensorSize_t i=0; i<logits->getDims()[0]; i++){
    auto y = (*yTrue)[i];
    auto s = sigmoid((*logits)[i]);

    auto g = s - y;
    res->set(g/bSize, i);
  }
  
  return {res};
}