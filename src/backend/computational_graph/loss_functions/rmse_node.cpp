/**
 * @file rmse_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "rmse_node.h"
#include "data_modeling/tensor_functions.h"

#include <iostream>

#ifdef __CUDA
#include "computational_graph/loss_functions/cuda/loss_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

vector< shared_ptr<Tensor> > RmseNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: 
    {
      const ftype bSize = yPred->getDims()[0];
      for(tensorSize_t i = 0; i < yPred->getSize(); i++){
        auto yi = (*yTrue)[i];
        auto yiHat = (*yPred)[i];

        auto denom = rmse * bSize + EPS_RMSE;
        auto g = (yiHat - yi) / denom;
        
        res->set(g, i);
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::rmseBackward(*res, *yPred, *yTrue, rmse);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
