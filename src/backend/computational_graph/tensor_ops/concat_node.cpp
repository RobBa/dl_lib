/**
 * @file concat_node.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "concat_node.h"

#ifdef __CUDA
#include "cuda_runtime.h"
#include "utility/cuda/cuda_common.cuh"
#endif

using namespace std;
using namespace cgraph;

std::vector<std::shared_ptr<Tensor>> ConcatNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  std::vector<std::shared_ptr<Tensor>> res;
  res.reserve(parents.size());

  tensorSize_t currSize = 0;
  for(const auto& p : parents) {
    res.emplace_back(make_shared<Tensor>(upstreamGrad.createView(p->getDims(), currSize)));
    currSize += p->getSize();
  }

  return res;
}