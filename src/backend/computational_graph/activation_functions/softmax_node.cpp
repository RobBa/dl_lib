/**
 * @file softmax_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-15
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "softmax_node.h"

#include "data_modeling/tensor_functions.h"

#include <iostream>

#ifdef __CUDA
#include "computational_graph/activation_functions/cuda/activation_nodes.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace cgraph;

/**
 * @brief Shape assumptions of softmax forward: [bsize, dim1, dim2, ..., stride]
 */
vector< shared_ptr<Tensor> > SoftmaxNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: {
      const tensorSize_t stride = yPred->getDims()[-1];
      tensorSize_t offset = 0;
      while(offset < yPred->getSize()) {
        for(tensorSize_t i = 0; i < stride; i++) {
          ftype grad = 0;
          const ftype yi = softmax->get(offset + i);

          for(tensorSize_t j = 0; j < stride; j++) {
            const ftype yj = softmax->get(offset + j);
            const ftype jacobian = (i == j) ? yi * (1 - yj) : -yi * yj;
            grad += upstreamGrad.get(offset + j) * jacobian;
          }
          res->set(grad, offset + i);
        }
        offset += stride;
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::softmaxBackward(*res, upstreamGrad, *softmax);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return {res};
}
