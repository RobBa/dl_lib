/**
 * @file bce_logits_loss.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "bce_sigmoid_loss.h"
#include "computational_graph/loss_functions/bce_sigmoid_node.h"

#include <cmath>

#ifdef __CUDA
#include "training/loss_functions/cuda/loss_functions.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace train;

/**
 * @brief Expected shapes: (batchsize) or (batchsize, 1)
 * @return Tensor of shape (1)
 */
shared_ptr<Tensor> BceSigmoidLoss::operator()(const shared_ptr<Tensor> y, const shared_ptr<Tensor> logits) const {
  if(!logits->getRequiresGrad()) {
    __throw_invalid_argument("logits must have gradient enabled");
  }
  else if(y->getDevice() != logits->getDevice()){
    __throw_invalid_argument("y and logits must be on same device");
  }
  else if(y->getDims()!=logits->getDims()){
    __throw_invalid_argument("Tensors must be of same shape");
  }

  shared_ptr<Tensor> res = nullptr;

  switch(y->getDevice()) {
    case Device::CPU: {
      auto bceSimplified = [](ftype y, ftype logit){
        constexpr ftype zero = 0;
        return std::max(logit, zero) - logit*y + log(1+exp(-std::abs(logit)));
      };

      const auto nBatches = y->getDims()[0];
      ftype loss = 0;
      for(tensorSize_t i=0; i<nBatches; i++){
        loss += bceSimplified((*y)[i], (*logits)[i]);
      }
      res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{loss / nBatches}, y->getDevice(), true);
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      res = make_shared<Tensor>(cuda_impl::bceSigmoidLoss(*y, *logits));
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  res->setCgNode(make_shared<cgraph::BceSigmoidNode>(y, logits));
  assert(res->getRequiresGrad());
  return res;
}
