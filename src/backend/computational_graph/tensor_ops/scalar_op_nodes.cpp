/**
 * @file scalar_op_nodes.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-17
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "scalar_op_nodes.h"

#include <utility>

using namespace std;
using namespace cgraph;

vector<shared_ptr<Tensor>> cgraph::ScalarAddNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  return {make_shared<Tensor>(upstreamGrad.createDeepCopy())};
}

vector<shared_ptr<Tensor>> cgraph::ScalarMulNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  auto res = make_shared<Tensor>(upstreamGrad.createDeepCopy());
  for(tensorSize_t i=0; i<res->getSize(); i++){
    res->setItem(res->getItem(i) * factor, i);
  }
  return {std::move(res)};
}