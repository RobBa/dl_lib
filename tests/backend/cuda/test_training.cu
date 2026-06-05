/**
 * @file test_training.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-06-05
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "data_modeling/tensor_functions.h"

#include "net_factories.h"

#include "training/optimizers/sgd.h"

#include "training/loss_functions/bce_sigmoid_loss.h"

using namespace std;

static ftype computeTotalNorm(const std::vector<std::shared_ptr<Tensor>>& params) {
  ftype norm = 0.0f;
  for (const auto& p : params) {
    auto grads = p->getGrads();
    if (!grads) continue;
    for (tensorSize_t i = 0; i < grads->getSize(); i++) {
      auto g = (*grads)[i];
      norm += g * g;
    }
  }
  return std::sqrt(norm);
}

TEST(CudaOptimizerTest, ZeroGrad_ClearsAllGradients) {
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, Device::CUDA, false);
  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0, 1.0, 1.0, 0.0}, Device::CUDA, false);

  auto net = makeBinaryNet(Device::CUDA);
  auto loss = make_shared<train::BceSigmoidLoss>();
  auto optim = make_shared<train::SgdOptimizer>(net->parameters(), 0.01f);

  auto pred = (*net)(x);
  auto l = (*loss)(y, pred);
  l->backward();

  bool anyNonZero = false;
  for(auto& p : net->parameters()) {
    if(p->getGrads()) {
      for(tensorSize_t i = 0; i < p->getGrads()->getSize(); i++) {
        if((*p->getGrads())[i] != 0.0f) {
          anyNonZero = true;
          break;
        }
      }
    }
  }
  EXPECT_TRUE(anyNonZero) << "Expected some non-zero gradients before zeroGrad";

  optim->zeroGrad();

  for(auto& p : net->parameters()) {
    if(p->getGrads()) {
      for(tensorSize_t i = 0; i < p->getGrads()->getSize(); i++) {
        ASSERT_NEAR((*p->getGrads())[i], 0.0f, 1e-5)
          << "Gradient not zeroed at index " << i;
      }
    }
  }
}

TEST(CudaOptimizerTest, GradClip_ScalesGradientsCorrectly) {
  // grads = [3, 4], L2 norm = 5; maxNorm = 2.5 → scale = 0.5
  auto param = TensorFunctions::makeSharedTensor({2}, {0.0, 0.0}, Device::CUDA, true);
  auto grads = TensorFunctions::makeSharedTensor({2}, {3.0f, 4.0f}, Device::CUDA, false);
  param->setGrads(grads);

  auto optim = make_shared<train::SgdOptimizer>(
    std::vector<std::shared_ptr<Tensor>>{param}, 0.01f);
  optim->clipGradients(2.5f);

  ASSERT_NEAR((*param->getGrads())[0], 1.5f, 1e-3f);
  ASSERT_NEAR((*param->getGrads())[1], 2.0f, 1e-3f);
}

TEST(CudaOptimizerTest, GradClip_BelowThreshold_NoChange) {
  auto param = TensorFunctions::makeSharedTensor({2}, {0.0, 0.0}, Device::CUDA, true);
  auto grads = TensorFunctions::makeSharedTensor({2}, {3.0f, 4.0f}, Device::CUDA, false);
  param->setGrads(grads);

  auto optim = make_shared<train::SgdOptimizer>(
    std::vector<std::shared_ptr<Tensor>>{param}, 0.01f);
  optim->clipGradients(1e6f);

  ASSERT_NEAR((*param->getGrads())[0], 3.0f, 1e-3f);
  ASSERT_NEAR((*param->getGrads())[1], 4.0f, 1e-3f);
}

TEST(CudaOptimizerTest, GradClip_NormIsEnforced) {
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, Device::CUDA, false);
  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0, 1.0, 1.0, 0.0}, Device::CUDA, false);

  auto net = makeBinaryNet(Device::CUDA);
  auto loss = make_shared<train::BceSigmoidLoss>();
  auto optim = make_shared<train::SgdOptimizer>(net->parameters(), 0.01f);

  auto pred = (*net)(x);
  auto l = (*loss)(y, pred);
  l->backward();

  const ftype maxNorm = 0.1f;
  optim->clipGradients(maxNorm);

  EXPECT_LE(computeTotalNorm(net->parameters()), maxNorm + 1e-3f);
}
