/**
 * @file sigmoid.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "sigmoid.h"
#include "computational_graph/activation_functions/sigmoid_node.h"

#include <cmath>

#ifdef __CUDA
#include "module/activation_functions/cuda/activations.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace module;

/**
 * @brief Sigmoid activation function.
 */
Tensor Sigmoid::operator()(const Tensor& t) const {
  auto res = t.createEmptyCopy();

  switch(t.getDevice()) {
    case Device::CPU: {
      constexpr ftype one = 1.0;
      auto compute = [](ftype x){
        if(x >= 0){
          return one / (one + exp(-x));
        }
        auto e = exp(x);
        return e / (one + e);
      };

      for(tensorSize_t i=0; i<t.getSize(); i++){
        res.set(compute(t[i]), i);
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::sigmoid(res, t);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return res;
}

shared_ptr<Tensor> Sigmoid::operator()(const shared_ptr<Tensor>& t) const {
  auto res = make_shared<Tensor>((*this)(*t));

  if(t->getRequiresGrad()){
    res->setCgNode(make_shared<cgraph::SigmoidNode>(t, res));
    assert(res->getRequiresGrad());
  }

  return res;
}
