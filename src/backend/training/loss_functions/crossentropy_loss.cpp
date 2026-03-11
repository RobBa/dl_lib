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

#include <cmath>

using namespace std;
using namespace train;

/**
 * @brief Expected shapes: (batch_size, n_classes)
 * @return Tensor of shape (1)
 */
shared_ptr<Tensor> CrossEntropyLoss::operator()(const Tensor& y, const shared_ptr<Tensor> & ypred) const {
  assert(ypred->getRequiresGrad());
  
  if(y.getDevice() != ypred->getDevice()){
    __throw_invalid_argument("y and ypred must be on same device");
  }
  else if(y.getDims()!=ypred->getDims()){
    __throw_invalid_argument("Tensors must be of same shape");
  }

  auto ce = [&y, &ypred](const tensorDim_t b){
    ftype res = 0;
    for(tensorDim_t i=0; i<y.getDims().getItem(-1); i++){
      res += y.getItem(b, i) * log(ypred->getItem(b, i));
    }
    return res;
  };

  const auto nBatches = y.getDims().getItem(0);
  ftype res = 0;
  for(tensorSize_t b=0; b<nBatches; b++){
    res += ce(b);
  }

  return make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{-res / nBatches}, y.getDevice(), true);;
}