/**
 * @file sigmoid_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "sigmoid_node.h"

#include <utility>

#ifdef __CUDA
#include "computational_graph/activation_functions/cuda/activation_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

vector<shared_ptr<Tensor>> SigmoidNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  auto res = make_shared<Tensor>(upstreamGrad.getDims(), upstreamGrad.getDevice(), false);

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: {
      auto derivative = [](ftype s){
        return s * (1-s);
      };

      for(tensorSize_t i=0; i<upstreamGrad.getSize(); i++){
        res->set(derivative((*sigmoid)[i]) * upstreamGrad[i], i);
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::sigmoidBackward(*res, upstreamGrad, *sigmoid);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
