/**
 * @file test_transformers.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-10-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "shared.h"

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

TEST(CudaSdpAttentionTest, ForwardMatchesHandComputedSoftmax) {
  // Same hand-computed case as the CPU test, run on CUDA tensors.
  auto Q = Tensor({2, 2}, {1.0, 0.0, 0.0, 1.0}, Device::CUDA);
  auto K = Tensor({2, 2}, {1.0, 0.0, 0.0, 1.0}, Device::CUDA);
  auto V = Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0}, Device::CUDA);

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);
  out.setDevice(Device::CPU);

  ASSERT_EQ(out.getDims(), Dimension({2, 2}));
  ASSERT_NEAR(out.get(0, 0), 1.6604769, 1e-4);
  ASSERT_NEAR(out.get(0, 1), 2.6604769, 1e-4);
  ASSERT_NEAR(out.get(1, 0), 2.3395231, 1e-4);
  ASSERT_NEAR(out.get(1, 1), 3.3395231, 1e-4);
}

TEST(CudaSdpAttentionTest, ForwardMatchesCpu) {
  // Cross-check CPU vs CUDA on a slightly larger, randomly initialised case.
  constexpr tensorDim_t seq = 5;
  constexpr tensorDim_t dk = 4;

  auto Qcpu = tensorfunct::Gaussian({seq, dk}, 1.0f);
  auto Kcpu = tensorfunct::Gaussian({seq, dk}, 1.0f);
  auto Vcpu = tensorfunct::Gaussian({seq, dk}, 1.0f);

  auto Qgpu = Qcpu.createDeepCopy(); Qgpu.setDevice(Device::CUDA);
  auto Kgpu = Kcpu.createDeepCopy(); Kgpu.setDevice(Device::CUDA);
  auto Vgpu = Vcpu.createDeepCopy(); Vgpu.setDevice(Device::CUDA);

  SdpAttentionFn attn;
  auto outCpu = attn(Kcpu, Qcpu, Vcpu, MaskType::None);
  auto outGpu = attn(Kgpu, Qgpu, Vgpu, MaskType::None);
  outGpu.setDevice(Device::CPU);

  ASSERT_EQ(outCpu.getDims(), outGpu.getDims());
  cuda_test::NumericalStabilityChecker checker(1e-4);
  for(tensorSize_t i = 0; i < outCpu.getSize(); i++) {
    checker.check(outCpu[i], outGpu[i], "index " + std::to_string(i));
  }
  checker.finalize();
}

TEST(CudaSdpAttentionTest, SingleTokenOutputEqualsV) {
  // With a single token, softmax over a 1-element row is always exactly 1,
  // regardless of Q/K, so attention degenerates to the identity on V.
  auto Q = tensorfunct::makeSharedTensor({1, 2}, {1.0, 2.0}, Device::CUDA, false);
  auto K = tensorfunct::makeSharedTensor({1, 2}, {5.0, -7.0}, Device::CUDA, false);
  auto V = tensorfunct::makeSharedTensor({1, 2}, {3.0, 4.0}, Device::CUDA, false);

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);
  out->setDevice(Device::CPU);

  ASSERT_NEAR(out->get(0), 3.0, 1e-4);
  ASSERT_NEAR(out->get(1), 4.0, 1e-4);
}

TEST(CudaSdpAttentionAutogradTest, SingleTokenGradientsAreExact) {
  // Continuing from SingleTokenOutputEqualsV: since softmax saturates to 1
  // for a single-element row, its Jacobian is exactly zero, so Q and K must
  // receive exactly zero gradient, while V receives exactly the upstream
  // gradient (output == V, unweighted).
  auto Q = tensorfunct::makeSharedTensor({1, 2}, {1.0, 2.0}, Device::CUDA, true);
  auto K = tensorfunct::makeSharedTensor({1, 2}, {5.0, -7.0}, Device::CUDA, true);
  auto V = tensorfunct::makeSharedTensor({1, 2}, {3.0, 4.0}, Device::CUDA, true);

  SdpAttentionFn attn;
  auto out = attn(K, Q, V, MaskType::None);
  auto loss = cgraph::sumTensor(out);
  loss->backward();

  auto vGrads = V->getGrads();
  ASSERT_NE(vGrads, nullptr);
  vGrads->setDevice(Device::CPU);
  ASSERT_NEAR((*vGrads)[0], 1.0, 1e-4);
  ASSERT_NEAR((*vGrads)[1], 1.0, 1e-4);

  auto qGrads = Q->getGrads();
  ASSERT_NE(qGrads, nullptr);
  qGrads->setDevice(Device::CPU);
  ASSERT_NEAR((*qGrads)[0], 0.0, 1e-4);
  ASSERT_NEAR((*qGrads)[1], 0.0, 1e-4);

  auto kGrads = K->getGrads();
  ASSERT_NE(kGrads, nullptr);
  kGrads->setDevice(Device::CPU);
  ASSERT_NEAR((*kGrads)[0], 0.0, 1e-4);
  ASSERT_NEAR((*kGrads)[1], 0.0, 1e-4);
}

TEST(CudaSdpAttentionAutogradTest, RequiresGradPropagatesOnlyFromInputsThatNeedIt) {
  auto Q = tensorfunct::makeSharedTensor({1, 2}, {1.0, 2.0}, Device::CUDA, true);
  auto K = tensorfunct::makeSharedTensor({1, 2}, {5.0, -7.0}, Device::CUDA, false);
  auto V = tensorfunct::makeSharedTensor({1, 2}, {3.0, 4.0}, Device::CUDA, false);

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

TEST(CudaMultiheadSelfAttnTest, ThrowsWhenHiddenDimNotDivisibleByHeads) {
  ASSERT_THROW(
    MultiheadSelfAttn(SelfAttentionType::ScaledDotProduct, /*nHeads=*/3, /*embeddingDim=*/8, /*hiddenDim=*/8),
    std::invalid_argument);
}

TEST(CudaMultiheadSelfAttnTest, ForwardOutputShape) {
  // See the CPU test for why the input's last dim must be hiddenDim/nHeads
  // and the output's last dim ends up nHeads*hiddenDim with the current wiring.
  constexpr tensorDim_t nHeads = 2;
  constexpr tensorDim_t hiddenDim = 4;
  constexpr tensorDim_t inputLastDim = hiddenDim / nHeads;

  MultiheadSelfAttn mha(SelfAttentionType::ScaledDotProduct, nHeads, /*embeddingDim=*/99, hiddenDim);

  auto input = tensorfunct::Ones({3, inputLastDim}, Device::CUDA);
  auto out = mha(input, MaskType::None);

  ASSERT_EQ(out.getDims(), Dimension({3, nHeads * hiddenDim}));
  ASSERT_EQ(out.getDevice(), Device::CUDA);
}

TEST(CudaMultiheadSelfAttnAutogradTest, BackwardPopulatesInputGradient) {
  constexpr tensorDim_t nHeads = 2;
  constexpr tensorDim_t hiddenDim = 4;
  constexpr tensorDim_t inputLastDim = hiddenDim / nHeads;

  MultiheadSelfAttn mha(SelfAttentionType::ScaledDotProduct, nHeads, /*embeddingDim=*/99, hiddenDim);

  auto input = tensorfunct::makeSharedTensor({3, inputLastDim},
    {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, Device::CUDA, true);

  auto out = mha(input, MaskType::None);
  ASSERT_TRUE(out->getRequiresGrad());

  auto loss = cgraph::sumTensor(out);
  loss->backward();

  auto grads = input->getGrads();
  ASSERT_NE(grads, nullptr);
  ASSERT_EQ(grads->getDims(), input->getDims());

  grads->setDevice(Device::CPU);
  for(tensorSize_t i = 0; i < grads->getSize(); i++) {
    ASSERT_FALSE(std::isnan((*grads)[i]));
    ASSERT_FALSE(std::isinf((*grads)[i]));
  }
}
