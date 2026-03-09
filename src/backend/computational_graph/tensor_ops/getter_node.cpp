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
  // upstreamGrad is scalar by definition
  assert(!upstreamGrad.getRequiresGrad() && upstreamGrad.getDims().nDims()==1);

  auto res = make_shared<Tensor>(parents[0]->getDims(), parents[0]->getDevice(), false);
  for(tensorSize_t i=0; i<res->getSize(); i++){
    res->setItem(0, i);
  }

  if(std::holds_alternative<tensorSize_t>(idx)){
    res->setItem(upstreamGrad.getItem(0), std::get<tensorSize_t>(idx));
  }
  else if(std::holds_alternative<multiDimIdx_t>(idx)){
    res->setItem(upstreamGrad.getItem(0), std::get<multiDimIdx_t>(idx));
  }
  else{
    __throw_runtime_error("Idx variant in unexpected state");
  }

  return { std::move(res) };
}