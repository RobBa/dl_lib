/**
 * @file sdp_attention.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "sdp_attention.h"
#include "module/activation_functions/softmax.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include <cmath>

using namespace module::transformer;
using namespace std;

Tensor SdpAttention::forward(const Tensor& K, const Tensor& Q, const Tensor& V, MaskEnum mask) {
  constexpr ftype nominator = 1.0f;
  const auto factor = nominator / sqrt(static_cast<ftype>(Q.getDims().get(-1)));
  auto scores = Q.matmul(K, false, true) * factor;

  auto softmax = Softmax();
  auto weights = softmax(scores);
  return weights.matmul(V, false, false);
}

shared_ptr<Tensor> SdpAttention::forward(const shared_ptr<Tensor>& K, const shared_ptr<Tensor>& Q, const shared_ptr<Tensor>& V, MaskEnum mask) {
  constexpr ftype nominator = 1.0f;
  const auto factor = nominator / sqrt(static_cast<ftype>(Q->getDims().get(-1)));
  auto scores = cgraph::mul(cgraph::matmul(Q, K, false, true), factor);

  auto softmax = Softmax();
  auto weights = softmax(scores);
  return cgraph::matmul(weights, V, false, false);
}