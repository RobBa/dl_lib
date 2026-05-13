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

vector< shared_ptr<Tensor> > SoftmaxNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  const auto& yPred = parents[0];
  auto res = make_shared<Tensor>(yPred->createEmptyCopy());

  switch(upstreamGrad.getDevice()) {
    case Device::CPU: {
      const auto bSize = yPred->getDims()[0];
      assert(bSize>0);

      for(tensorDim_t b=0; b<bSize; b++){
        for(tensorDim_t i=0; i<yPred->getDims()[1]; i++){
          ftype grad = 0;
          const ftype yi = softmax->get(b, i);

          for(tensorDim_t j=0; j<yPred->getDims()[1]; j++){
            ftype yj = softmax->get(b, j);
            ftype jacobian = (i==j) ? yi*(1-yj) : -yi*yj;
            grad += upstreamGrad.get(b, j) * jacobian;
          }
          res->set(grad, b, i);
        }
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
