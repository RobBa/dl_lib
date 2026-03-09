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

#include "data_modeling/tensor_functions.h"

using namespace std;
using namespace graph;

vector< shared_ptr<Tensor> > AddNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());
  auto weightGrad = make_shared<Tensor>(upstreamGrad.createDeepCopy());
  
  if(broadcasted){
    auto biasGrad = make_shared<Tensor>(TensorFunctions::SumOverDims(*weightGrad));
    return {weightGrad, biasGrad};
  }
  
  return {weightGrad, weightGrad};
}