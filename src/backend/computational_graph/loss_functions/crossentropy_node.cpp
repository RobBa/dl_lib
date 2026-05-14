/**
 * @file crossentropy_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-02-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "crossentropy_node.h"
#include "data_modeling/tensor_functions.h"

#ifdef __CUDA
#include "computational_graph/loss_functions/cuda/loss_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

/**
 * @brief Backward function on crossentropy-node. Uses cached values of forward pass 
 * for higher efficiency.
 */
vector< shared_ptr<Tensor> > CrossEntropyNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: {
      const tensorSize_t stride = yPred->getDims()[-1];
      const ftype bSize = static_cast<ftype>(yPred->getSize() / stride);
      for(tensorSize_t i = 0; i < yPred->getSize(); i++) {
        auto g = -(*yTrue)[i] / std::max((*yPred)[i], EPS_CROSSENTROPY);
        res->set(g / bSize, i);
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::crossEntropyBackward(*res, *yPred, *yTrue);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
