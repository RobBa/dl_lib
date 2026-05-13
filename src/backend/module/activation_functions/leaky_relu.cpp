/**
 * @file leaky_relu.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "leaky_relu.h"
#include "computational_graph/activation_functions/leaky_relu_node.h"

#ifdef __CUDA
#include "module/activation_functions/cuda/activations.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace module;

Tensor LeakyReLu::operator()(const Tensor& t) const {
  auto res = t.createDeepCopy();

  switch(t.getDevice()) {
    case Device::CPU:
      for(tensorSize_t i=0; i<t.getSize(); i++){
        res.set(std::max(t[i], t[i]*eps), i);
      }
      break;
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::leakyRelu(res, t, eps);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return res;
}

shared_ptr<Tensor> LeakyReLu::operator()(const shared_ptr<Tensor>& t) const {
  auto res = make_shared<Tensor>((*this)(*t));
  
  if(t->getRequiresGrad()){
    res->setCgNode(make_shared<cgraph::LeakyReLuNode>(t, eps));
    assert(res->getRequiresGrad());
  }

  return res;  
}