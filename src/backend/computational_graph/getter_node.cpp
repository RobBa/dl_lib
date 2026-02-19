/**
 * @file getter_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-18
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "getter_node.h"

using namespace std;
using namespace graph;

vector< shared_ptr<Tensor> > GetterNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  return { make_shared<Tensor>(upstreamGrad.createDeepCopy()) };
}