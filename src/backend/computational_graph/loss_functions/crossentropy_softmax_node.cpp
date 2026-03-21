/**
 * @file crossentropy_softmax_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-17
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "crossentropy_softmax_node.h"

#include "module/activation_functions/softmax.h"

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > CrossEntropySoftmaxNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& logits = parents[0];
  auto res = make_shared<Tensor>(logits->createEmptyCopy());

  const auto softmax = module::Softmax();
  const auto s = softmax(*logits);

  ftype bSize = logits->getDims()[0];
  for(tensorSize_t b=0; b<logits->getDims()[0]; b++){
    for(tensorSize_t i=0; i<logits->getDims()[1]; i++){
      auto g = s.get(b, i) - yTrue->get(b, i);
      res->set(g / bSize, b, i);
    }
  }

  return {res};
}