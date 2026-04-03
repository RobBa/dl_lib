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

#include "bce_loss.h"

#include "computational_graph/loss_functions/bce_node.h"

#include <cmath>

using namespace std;
using namespace train;

/**
 * @brief Expected shapes: (batchsize) or (batchsize, 1)
 * @return Tensor of shape (1)
 */
shared_ptr<Tensor> BceLoss::operator()(const shared_ptr<Tensor> y, const shared_ptr<Tensor> ypred) const {
  if(!ypred->getRequiresGrad()) {
    __throw_invalid_argument("ypred must have gradient enabled");
  }  
  else if(y->getDevice() != ypred->getDevice()){
    __throw_invalid_argument("y and ypred must be on same device");
  }
  else if(y->getDims()!=ypred->getDims()){
    __throw_invalid_argument("Tensors must be of same shape");
  }

  auto bce = [](ftype y, ftype ypred){
    return y*log(std::max(ypred, EPS_BCE)) + (1-y)*log(std::max(1-ypred, EPS_BCE));
  };

  const auto nBatches = y->getDims()[0];

  ftype loss = 0;
  for(tensorSize_t i=0; i<nBatches; i++){
    loss += bce((*y)[i], (*ypred)[i]);
  }

  auto res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{-loss / nBatches}, y->getDevice(), true);
  res->setCgNode(make_shared<cgraph::BceNode>(y, ypred));
  assert(res->getRequiresGrad());

  return res; 
}