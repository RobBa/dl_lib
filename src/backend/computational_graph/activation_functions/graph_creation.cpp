/**
 * @file graph_creation.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "graph_creation.h"

#include "relu_node.h"
#include "leaky_relu_node.h"

using namespace std;
using namespace activation;

shared_ptr<Tensor> doActivation(const ReLu& r, const shared_ptr<Tensor>& t) {
  auto res = make_shared<Tensor>(r(*t));
  if(t->getRequiresGrad()){
    res->setCgNode(make_shared<graph::ReLuNode>(t));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> doActivation(const LeakyReLu& r, const shared_ptr<Tensor>& t) {
  auto res = make_shared<Tensor>(r(*t));
  if(t->getRequiresGrad()){
    res->setCgNode(make_shared<graph::LeakyReLuNode>(t, r.getEps()));
    assert(res->getRequiresGrad());
  }
  return res;
}