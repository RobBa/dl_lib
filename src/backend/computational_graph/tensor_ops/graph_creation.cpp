/**
 * @file graph_creation.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-21
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "graph_creation.h"
#include "data_modeling/tensor_functions.h"

#include "concat_node.h"

#include <algorithm>

using namespace std;

/**
 * @brief Autograd enabled version of concat(). For more info read the overload's docstring.
 */
std::shared_ptr<Tensor> cgraph::concat(const std::vector< std::shared_ptr<Tensor> >& input, int dim) {
#ifndef NDEBUG
  if(input.size() < 2) {
    throw std::invalid_argument("Concat must have at least two tensors");
  }
#endif

  if(dim < 0) {
    dim += input.at(0)->getDims().nDims();
    assert(dim >= 0);
  }

  std::vector<const Tensor*> ptrs;
  ptrs.reserve(input.size());

  for (const auto& t : input) {
    ptrs.push_back(t.get());
  }

  auto res = std::make_shared<Tensor>(tensorfunct::concatImpl(ptrs, dim));

  const bool requiresGrad = std::any_of(input.cbegin(), input.cend(), 
                                        [](const auto& x) {
                                          return x->getRequiresGrad(); }
                                       );

  if(requiresGrad) {
    res->setCgNode(std::make_shared<cgraph::ConcatNode>(input, dim));
    assert(res->getRequiresGrad());
  }
  
  return res;
}