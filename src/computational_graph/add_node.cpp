/**
 * @file add_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "add_node.h"
#include "tensor.h"

using namespace std;
using namespace graph;

vector< shared_ptr<Tensor> > AddNode::backward(const Tensor& upstreamGrad) {
  auto res = make_shared<Tensor>(upstreamGrad.createDeepCopy());
  return {res, res}; // TODO: make sure that this works as intended
}