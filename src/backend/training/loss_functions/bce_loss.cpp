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

#include <cmath>

using namespace std;
using namespace train;

/**
 * @brief Expected shapes: (batch_size)
 * @return Tensor of shape (1)
 */
shared_ptr<Tensor> BceLoss::operator()(const shared_ptr<Tensor>& y, const shared_ptr<Tensor>& ypred) const {
  assert(ypred->getRequiresGrad());
  
  if(y->getDevice() != ypred->getDevice()){
    __throw_invalid_argument("y and ypred must be on same device");
  }
  else if(y->getDims()!=ypred->getDims()){
    __throw_invalid_argument("Tensors must be of same shape");
  }

  auto bce = [](ftype y, ftype ypred){
    return y*log(ypred) + (1-y)*log(1-ypred);
  };

  const auto nBatches = y->getDims().getItem(0);

  ftype res = 0;
  for(tensorSize_t i=0; i<nBatches; i++){
    res += bce((*y)[i], (*ypred)[i]);
  }

  return make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{-res / nBatches}, y->getDevice(), true);;
}