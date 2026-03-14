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

using namespace std;
using namespace module;

/**
 * @brief Sigmoid activation function.
 */
Tensor Sigmoid::operator()(const Tensor& t) const {
  auto res = t.createEmptyCopy();

  auto compute = [](ftype x){
    if(x>=0){
      return static_cast<ftype>(1.0f) / (static_cast<ftype>(1.0f) + exp(x));
    }
    auto e = exp(x);
    return e / (static_cast<ftype>(1.0f) + e);
  };

  for(tensorSize_t i=0; i<t.getSize(); i++){
    res.setItem(compute(t[i]), i);
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
