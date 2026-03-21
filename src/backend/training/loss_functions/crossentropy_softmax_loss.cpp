/**
 * @file crossentropy_logits_loss.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelflogits->nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-17
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "crossentropy_softmax_loss.h"

#include "computational_graph/loss_functions/crossentropy_softmax_node.h"

#include <cmath>

using namespace std;
using namespace train;

/**
 * @brief Expected shapes: (batch_size, n_classes)
 * @return Tensor of shape (1)
 */
shared_ptr<Tensor> CrossEntropySoftmaxLoss::operator()(const shared_ptr<Tensor> y, const shared_ptr<Tensor> logits) const {
  if(!logits->getRequiresGrad()) {
    __throw_invalid_argument("logits must have gradient enabled");
  }
  else if(y->getDevice() != logits->getDevice()){
    __throw_invalid_argument("y and logits must be on same device");
  }
  else if(y->getDims()!=logits->getDims()){
    __throw_invalid_argument("Tensors must be of same shape");
  }

  ////////////////////////////////////////////////

  const auto nRows = logits->getDims()[-2];
  const auto nCols = logits->getDims()[-1];

  // pre-compute exponents and max-values
  vector<ftype> maxValues(nRows); 
  Tensor tmp(logits->getDims(), logits->getDevice(), false);
  for(tensorDim_t i=0; i<nRows; i++){
    // for numerical stability, avoid inf
    ftype maxV = -std::numeric_limits<ftype>::infinity();
    for(tensorDim_t j=0; j<nCols; j++){
      maxV = std::max(maxV, logits->get(i, j));
    }

    maxValues[i] = maxV;

    for(tensorDim_t j=0; j<nCols; j++){
      ftype e = logits->get(i, j)-maxV;
      tmp.set(exp(e), i, j);
    }
  }

  const tensorSize_t stride = logits->getDims()[-1];
  ftype loss = 0;

  /** 
   * CE = -sum_i(y_i * z_i) + log(sum_j(exp(z_j))) with 
   * log(sum_j(exp(z_j))) = max(z) + log(sum_j(exp(z_j - max(z)))).
   * for numerical stability 
   */ 
  auto compute = [&loss, &y, &logits, &tmp, &maxValues, stride](tensorSize_t start){
    ftype lsum = 0;
    for(tensorSize_t i=start; i<start+stride; i++){
      lsum += tmp[i];
    }
    lsum = log(lsum);

    const tensorSize_t j = start/stride;
    for(tensorSize_t i=start; i<start+stride; i++){
      if((*y)[i]>0){ // y either zero or one
        loss += -(*logits)[i] + maxValues[j] + lsum;
      }
    }
  };
  
  tensorSize_t offset=0;
  while(offset<logits->getSize()) {
    compute(offset);
    offset += stride;
  }

  auto res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{loss / logits->getDims()[0]}, y->getDevice(), true);
  res->setCgNode(std::make_shared<cgraph::CrossEntropySoftmaxNode>(y, logits));
  assert(res->getRequiresGrad());

  return res;
}