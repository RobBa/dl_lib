/**
 * @file softmax.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "softmax.h"
#include "computational_graph/activation_functions/softmax_node.h"

#include <cmath>

#ifdef __CUDA
#include "module/activation_functions/cuda/activations.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace module;

/**
 * @brief Softmax over last dimension. Expects shape
 * (dim1, dim2, ..., n_classes)
 * @return Tensor of shape (dim1, dim2, ..., n_classes) [== input.shape]
 */
Tensor Softmax::operator()(const Tensor& t) const {
  if(t.getDims().nDims()<2){
    __throw_invalid_argument("Softmax expects input shape of minimum two dimensions");
  }

  auto res = t.createEmptyCopy();

  switch(t.getDevice()) {
    case Device::CPU: {
      const tensorSize_t stride = t.getDims()[-1];

      // pre-compute exponents, centering each slice around its max for numerical stability
      Tensor tmp(t.getDims(), t.getDevice(), false);
      tensorSize_t offset = 0;
      while(offset < t.getSize()) {
        ftype maxValue = -std::numeric_limits<ftype>::infinity();
        for(tensorSize_t i = offset; i < offset + stride; i++) {
          maxValue = std::max(maxValue, t[i]);
        }

        for(tensorSize_t i = offset; i < offset + stride; i++) {
          tmp.set(exp(t[i] - maxValue), i);
        }

        offset += stride;
      }

      auto compute = [&res, &tmp, stride](tensorSize_t start){
        ftype sum = 0;
        for(tensorSize_t i = start; i < start+stride; i++){
          sum += tmp[i];
        }

        for(tensorSize_t i = start; i < start + stride; i++){
          res.set(tmp[i] / sum, i);
        }
      };

      offset = 0;
      while(offset < res.getSize()) {
        compute(offset);
        offset += stride;
      }
      break;
    }
    case Device::CUDA:
    #ifdef __CUDA
      cuda_impl::softmax(res, t);
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
    #endif
      break;
  }

  return res;
}

shared_ptr<Tensor> Softmax::operator()(const shared_ptr<Tensor>& t) const {
  auto res = make_shared<Tensor>((*this)(*t));
  
  if(t->getRequiresGrad()){
    res->setCgNode(make_shared<cgraph::SoftmaxNode>(t, res));
    assert(res->getRequiresGrad());
  }

  return res;  
}
