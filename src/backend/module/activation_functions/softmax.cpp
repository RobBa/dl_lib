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

#include <cmath>

using namespace std;
using namespace module;

/**
 * @brief Softmax over last dimension. Expects shape
 * (dim1, dim2, ..., n_classes)
 * @return Tensor of shape (dim1, dim2, ..., n_classes) [== input.shape]
 */
Tensor Softmax::operator()(const Tensor& t) const {
  Tensor res(t.getDims(), t.getDevice());

  Tensor tmp(t.getDims(), t.getDevice());
  for(tensorSize_t i=0; i<t.getSize(); i++){
    tmp.set(static_cast<ftype>(exp(t[i])), i);
  }

  const tensorSize_t stride = t.getDims()[-1];
  auto compute = [&t, &res, &tmp, stride](tensorSize_t start){
    ftype sum = 0;
    for(tensorSize_t i=0; i<stride; i++){
      sum += static_cast<ftype>(t[start+i]);
    }

    for(tensorSize_t i=0; i<t.getSize(); i++){
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
    //res->setCgNode(make_shared<cgraph::LeakyReLuNode>(t, eps));
    assert(res->getRequiresGrad());
  }

  return res;  
}
