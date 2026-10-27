/**
 * @file test_transformers.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-10-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "module/transformer/attention/sdp_attention_fn.h"
#include "module/transformer/attention/multihead_self_attention.h"
#include "module/transformer/params/params.h"

#include "data_modeling/tensor_functions.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include <cmath>

using namespace std;
using namespace module;
using namespace module::transformer;

/************************************************************************************
 ************************************ SdpAttentionFn *********************************
 ***********************************************************************************/

TEST(SdpAttentionTest, ForwardMatchesHandComputedSoftmax) {
  // Q = K = I(2), V = [[1,2],[3,4]], dk = 2.
  // scores = Q@K^T = I, scaled by 1/sqrt(2) => softmax rows = [0.66976, 0.33024] (and mirrored)
  // out = weights @ V, computed independently in python and cross-checked here.
  auto Q = Tensor({2, 2}, {1.0, 0.0, 0.0, 1.0});
  auto K = Tensor({2, 2}, {1.0, 0.0, 0.0, 1.0});
  auto V = Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0});

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);

  ASSERT_EQ(out.getDims(), Dimension({2, 2}));
  ASSERT_NEAR(out.get(0, 0), 1.6604769, 1e-4);
  ASSERT_NEAR(out.get(0, 1), 2.6604769, 1e-4);
  ASSERT_NEAR(out.get(1, 0), 2.3395231, 1e-4);
  ASSERT_NEAR(out.get(1, 1), 3.3395231, 1e-4);
}

TEST(SdpAttentionTest, ForwardBatchedMatchesPerBatchSlice) {
  // Two identical batches of the ForwardMatchesHandComputedSoftmax case: batching
  // must not change the per-slice result.
  auto Q = Tensor({2, 2, 2}, {1.0, 0.0, 0.0, 1.0,
                              1.0, 0.0, 0.0, 1.0});
  auto K = Tensor({2, 2, 2}, {1.0, 0.0, 0.0, 1.0,
                              1.0, 0.0, 0.0, 1.0});
  auto V = Tensor({2, 2, 2}, {1.0, 2.0, 3.0, 4.0,
                              1.0, 2.0, 3.0, 4.0});

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);

  ASSERT_EQ(out.getDims(), Dimension({2, 2, 2}));
  for(int b = 0; b < 2; b++) {
    ASSERT_NEAR(out.get(b, 0, 0), 1.6604769, 1e-4);
    ASSERT_NEAR(out.get(b, 0, 1), 2.6604769, 1e-4);
    ASSERT_NEAR(out.get(b, 1, 0), 2.3395231, 1e-4);
    ASSERT_NEAR(out.get(b, 1, 1), 3.3395231, 1e-4);
  }
}

TEST(SdpAttentionTest, SingleTokenOutputEqualsV) {
  // With a single token, softmax over a 1-element row is always exactly 1,
  // regardless of Q/K, so attention degenerates to the identity on V.
  auto Q = tensorfunct::makeSharedTensor({1, 2}, {1.0, 2.0}, false);
  auto K = tensorfunct::makeSharedTensor({1, 2}, {5.0, -7.0}, false);
  auto V = tensorfunct::makeSharedTensor({1, 2}, {3.0, 4.0}, false);

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);

  ASSERT_NEAR(out->get(0), 3.0, 1e-5);
  ASSERT_NEAR(out->get(1), 4.0, 1e-5);
}

TEST(SdpAttentionAutogradTest, SingleTokenGradientsAreExact) {
  // Continuing from SingleTokenOutputEqualsV: since softmax saturates to 1
  // for a single-element row, its Jacobian is exactly zero, so Q and K must
  // receive exactly zero gradient, while V receives exactly the upstream
  // gradient (output == V, unweighted).
  auto Q = tensorfunct::makeSharedTensor({1, 2}, {1.0, 2.0}, true);
  auto K = tensorfunct::makeSharedTensor({1, 2}, {5.0, -7.0}, true);
  auto V = tensorfunct::makeSharedTensor({1, 2}, {3.0, 4.0}, true);

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);
  auto loss = cgraph::sumTensor(out);
  loss->backward();

  ASSERT_NE(V->getGrads(), nullptr);
  ASSERT_NEAR((*V->getGrads())[0], 1.0, 1e-5);
  ASSERT_NEAR((*V->getGrads())[1], 1.0, 1e-5);

  ASSERT_NE(Q->getGrads(), nullptr);
  ASSERT_NEAR((*Q->getGrads())[0], 0.0, 1e-5);
  ASSERT_NEAR((*Q->getGrads())[1], 0.0, 1e-5);

  ASSERT_NE(K->getGrads(), nullptr);
  ASSERT_NEAR((*K->getGrads())[0], 0.0, 1e-5);
  ASSERT_NEAR((*K->getGrads())[1], 0.0, 1e-5);
}

TEST(SdpAttentionAutogradTest, RequiresGradPropagatesOnlyFromInputsThatNeedIt) {
  auto Q = tensorfunct::makeSharedTensor({1, 2}, {1.0, 2.0}, true);
  auto K = tensorfunct::makeSharedTensor({1, 2}, {5.0, -7.0}, false);
  auto V = tensorfunct::makeSharedTensor({1, 2}, {3.0, 4.0}, false);

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);

  ASSERT_TRUE(out->getRequiresGrad());

  auto loss = cgraph::sumTensor(out);
  loss->backward();

  ASSERT_NE(Q->getGrads(), nullptr);
  ASSERT_EQ(K->getGrads(), nullptr);
  ASSERT_EQ(V->getGrads(), nullptr);
}

/************************************************************************************
 ********************************* MultiheadSelfAttn *********************************
 ***********************************************************************************/

TEST(MultiheadSelfAttnTest, ThrowsWhenHiddenDimNotDivisibleByHeads) {
  ASSERT_THROW(
    MultiheadSelfAttn(SelfAttentionType::ScaledDotProduct, /*nHeads=*/3, /*embeddingDim=*/8, /*hiddenDim=*/8),
    std::invalid_argument);
}

TEST(MultiheadSelfAttnTest, ForwardOutputShape) {
  // Each head's K/Q/V currently maps (hiddenDim / nHeads) -> hiddenDim (see note
  // in the PR description / chat: the constructor's own embeddingDim argument is
  // never actually used), so the input's last dim must be hiddenDim/nHeads, and
  // concatenating nHeads of these hiddenDim-wide outputs gives nHeads*hiddenDim.
  constexpr tensorDim_t nHeads = 2;
  constexpr tensorDim_t hiddenDim = 4;
  constexpr tensorDim_t inputLastDim = hiddenDim / nHeads;

  MultiheadSelfAttn mha(SelfAttentionType::ScaledDotProduct, nHeads, /*embeddingDim=*/99, hiddenDim);

  auto input = tensorfunct::Ones({3, inputLastDim});
  auto out = mha(input, MaskType::None);

  ASSERT_EQ(out.getDims(), Dimension({3, nHeads * hiddenDim}));
}

TEST(MultiheadSelfAttnAutogradTest, BackwardPopulatesInputGradient) {
  constexpr tensorDim_t nHeads = 2;
  constexpr tensorDim_t hiddenDim = 4;
  constexpr tensorDim_t inputLastDim = hiddenDim / nHeads;

  MultiheadSelfAttn mha(SelfAttentionType::ScaledDotProduct, nHeads, /*embeddingDim=*/99, hiddenDim);

  auto input = tensorfunct::makeSharedTensor({3, inputLastDim},
    {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, true);

  auto out = mha(input, MaskType::None);
  ASSERT_TRUE(out->getRequiresGrad());

  auto loss = cgraph::sumTensor(out);
  loss->backward();

  auto grads = input->getGrads();
  ASSERT_NE(grads, nullptr);
  ASSERT_EQ(grads->getDims(), input->getDims());
  for(tensorSize_t i = 0; i < grads->getSize(); i++) {
    ASSERT_FALSE(std::isnan((*grads)[i]));
    ASSERT_FALSE(std::isinf((*grads)[i]));
  }
}
