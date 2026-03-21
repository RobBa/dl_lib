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

  const auto nRows = t.getDims()[-2];
  const auto nCols = t.getDims()[-1];

  // pre-compute exponents
  Tensor tmp(t.getDims(), t.getDevice(), false);
  for(tensorDim_t i=0; i<nRows; i++){
    // for numerical stability, avoid large values
    // by centering around maxValue for each sample
    ftype maxValue = -std::numeric_limits<ftype>::infinity();
    for(tensorDim_t j=0; j<nCols; j++){
      maxValue = std::max(maxValue, t.get(i, j));
    }

    for(tensorDim_t j=0; j<nCols; j++){
      ftype e = t.get(i, j)-maxValue;
      tmp.set(exp(e), i, j);
    }
  }

  const tensorSize_t stride = t.getDims()[-1];
  Tensor res(t.getDims(), t.getDevice());
  auto compute = [&res, &tmp, stride](tensorSize_t start){
    ftype sum = 0;
    for(tensorSize_t i=start; i<start+stride; i++){
      sum += tmp[i];
    }

    for(tensorSize_t i=start; i<start+stride; i++){
      res.set(tmp[i] / sum, i);
    }
  };
  
  tensorSize_t offset=0;
  while(offset<res.getSize()) {
    compute(offset);
    offset += stride;
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
