/**
 * @file leaky_relu_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-07
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "leaky_relu_node.h"

#include <utility>

#ifdef __CUDA
#include "computational_graph/activation_functions/cuda/activation_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

vector<shared_ptr<Tensor>> LeakyReLuNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  auto res = make_shared<Tensor>(upstreamGrad.getDims(), upstreamGrad.getDevice(), false);
  const auto& parent = parents[0];

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: {
      constexpr ftype zero = 0.0;
      for(tensorSize_t i = 0; i < upstreamGrad.getSize(); i++){
        res->set((*parent)[i] > zero ? upstreamGrad[i] : upstreamGrad[i] * eps, i);
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::leakyReluBackward(*res, upstreamGrad, *parent, eps);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
