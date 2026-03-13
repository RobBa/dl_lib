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

using namespace std;
using namespace module;

Tensor LeakyReLu::operator()(const Tensor& t) const {
  auto res = t.createDeepCopy();

  for(tensorSize_t i=0; i<t.getSize(); i++){
    constexpr ftype zero = 0;
    if(t[i] < zero){
      res.setItem(eps, i);
    }
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