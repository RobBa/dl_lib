/**
 * @file crossentropy_loss.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "crossentropy_loss.h"
#include "computational_graph/loss_functions/crossentropy_node.h"

#include <cmath>

#ifdef __CUDA
#include "training/loss_functions/cuda/loss_functions.cuh"
#else
#include <stdexcept>
#endif

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

  shared_ptr<Tensor> res = nullptr;

  switch(y->getDevice()) {
    case Device::CPU: {
      const tensorSize_t stride = y->getDims()[-1];
      const tensorSize_t nSamples = y->getSize() / stride;

      ftype loss = 0;
      for (tensorSize_t i = 0; i < y->getSize(); i++) {
        loss += (*y)[i] * log(std::max((*ypred)[i], EPS_CROSSENTROPY));
      }

      res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{-loss / nSamples}, y->getDevice(), true);
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      res = make_shared<Tensor>(vector<tensorDim_t>{1}, Device::CUDA, true);
      cuda_impl::crossEntropyLoss(*res, *y, *ypred);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  res->setCgNode(std::make_shared<cgraph::CrossEntropyNode>(y, ypred));
  assert(res->getRequiresGrad());
  return res;
}
