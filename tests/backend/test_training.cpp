/**
 * @file test_training.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-06-05
 *
 * @copyright Copyright (c) 2026
 *
 */

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

TEST(OptimizerTest, ZeroGrad_ClearsAllGradients) {
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, false);
  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0, 1.0, 1.0, 0.0}, false);

  auto net = makeBinaryNet();
  auto loss = std::make_shared<train::BceSigmoidLoss>();
  auto optim = std::make_shared<train::SgdOptimizer>(
    net->parameters(), 0.01f);

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

TEST(OptimizerTest, GradClip_ScalesGradientsCorrectly) {
  // grads = [3, 4], L2 norm = 5; maxNorm = 2.5 → scale = 0.5
  auto param = TensorFunctions::makeSharedTensor({2}, {0.0, 0.0}, true);
  auto grads = TensorFunctions::makeSharedTensor({2}, {3.0f, 4.0f}, false);
  param->setGrads(grads);

  auto optim = std::make_shared<train::SgdOptimizer>(
    std::vector<std::shared_ptr<Tensor>>{param}, 0.01f);
  optim->clipGradients(2.5f);

  ASSERT_NEAR((*param->getGrads())[0], 1.5f, 1e-4f);
  ASSERT_NEAR((*param->getGrads())[1], 2.0f, 1e-4f);
}

TEST(OptimizerTest, GradClip_BelowThreshold_NoChange) {
  auto param = TensorFunctions::makeSharedTensor({2}, {0.0, 0.0}, true);
  auto grads = TensorFunctions::makeSharedTensor({2}, {3.0f, 4.0f}, false);
  param->setGrads(grads);

  auto optim = std::make_shared<train::SgdOptimizer>(
    std::vector<std::shared_ptr<Tensor>>{param}, 0.01f);
  optim->clipGradients(1e6f);

  ASSERT_NEAR((*param->getGrads())[0], 3.0f, 1e-5f);
  ASSERT_NEAR((*param->getGrads())[1], 4.0f, 1e-5f);
}

TEST(OptimizerTest, GradClip_NormIsEnforced) {
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, false);
  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0, 1.0, 1.0, 0.0}, false);

  auto net = makeBinaryNet();
  auto loss = std::make_shared<train::BceSigmoidLoss>();
  auto optim = std::make_shared<train::SgdOptimizer>(net->parameters(), 0.01f);

  auto pred = (*net)(x);
  auto l = (*loss)(y, pred);
  l->backward();

  const ftype maxNorm = 0.1f;
  optim->clipGradients(maxNorm);

  EXPECT_LE(computeTotalNorm(net->parameters()), maxNorm + 1e-4f);
}
