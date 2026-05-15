/**
 * @file bce_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "bce_node.h"
#include "data_modeling/tensor_functions.h"

#ifdef __CUDA
#include "computational_graph/loss_functions/cuda/loss_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > BceNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: {
      ftype bSize = yPred->getDims()[0];
      for(tensorSize_t i = 0; i < yPred->getDims()[0]; i++){
        auto yi = (*yTrue)[i];
        auto yiHat = (*yPred)[i];
        auto g = -yi / std::max(yiHat, EPS_BCE) + (1 - yi) / std::max(1 - yiHat, EPS_BCE);
        res->set(g / bSize, i);
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::bceBackward(*res, *yPred, *yTrue);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
