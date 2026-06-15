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

#ifdef __CUDA
#include "computational_graph/loss_functions/cuda/loss_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > CrossEntropySoftmaxNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad() && parents[0]->getDims().nDims() == 2);

  const auto& logits = parents[0];
  auto res = make_shared<Tensor>(logits->createEmptyCopy());

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: 
    {
      static const auto softmax = module::Softmax();
      const auto s = softmax(*logits);

      const tensorSize_t stride = logits->getDims().get(-1);
      const tensorSize_t bSize = logits->getSize() / stride;

      for(tensorSize_t b = 0; b < bSize; b++){
        for(tensorSize_t i = 0; i < stride; i++){
          const tensorSize_t flatIdx = b * stride + i;
          auto g = s[flatIdx] - (*yTrue)[flatIdx];
          res->set(g / bSize, flatIdx);
        }
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::crossEntropySoftmaxBackward(*res, *logits, *yTrue);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
