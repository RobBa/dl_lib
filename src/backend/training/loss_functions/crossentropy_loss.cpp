/**
 * @file bce_loss.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "crossentropy_loss.h"

#include "computational_graph/loss_functions/crossentropy_node.h"

#include <cmath>

using namespace std;
using namespace train;

/**
 * @brief Expected shapes: (batch_size, n_classes)
 * @return Tensor of shape (1)
 */
shared_ptr<Tensor> CrossEntropyLoss::operator()(const shared_ptr<Tensor> y, const shared_ptr<Tensor> ypred) const {
  if(!ypred->getRequiresGrad()) {
    __throw_invalid_argument("ypred must have gradient enabled");
  }
  else if(y->getDevice() != ypred->getDevice()){
    __throw_invalid_argument("y and ypred must be on same device");
  }
  else if(y->getDims()!=ypred->getDims()){
    __throw_invalid_argument("Tensors must be of same shape");
  }

  auto ce = [&y, &ypred](const tensorDim_t b){
    ftype res = 0;
    for(tensorDim_t i=0; i<y->getDims()[-1]; i++){
      constexpr ftype eps = 1e-6;
      res += y->get(b, i) * log(std::max(ypred->get(b, i), eps));
    }
    return res;
  };

  const auto nBatches = y->getDims()[0];
  ftype loss = 0;
  for(tensorSize_t b=0; b<nBatches; b++){
    loss += ce(b);
  }

  auto res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{-loss / nBatches}, y->getDevice(), true);
  res->setCgNode(std::make_shared<cgraph::CrossEntropyNode>(y, ypred));
  assert(res->getRequiresGrad());

  return res;
}