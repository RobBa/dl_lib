/**
 * @file test_losses.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "data_modeling/tensor_functions.h"

#include "training/loss_functions/rmse_loss.h"
#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include <cmath>

using namespace train;

static constexpr ftype delta = 1e-4f;

TEST(CudaLossTest, CrossEntropyForward) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0}, Device::CUDA, false);

  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 3}, {0.7, 0.2, 0.1,
         0.1, 0.8, 0.1}, Device::CUDA, true);

  CrossEntropyLoss loss;
  auto result = loss(y, ypred);

  const ftype expected = -(std::log(0.7f) + std::log(0.8f)) / 2.0f;
  ASSERT_NEAR((*result)[0], expected, delta);
}

TEST(CudaLossTest, CrossEntropyPerfectPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0}, Device::CUDA, false);

  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 3}, {0.999, 0.0005, 0.0005,
         0.0005, 0.999, 0.0005}, Device::CUDA, true);

  CrossEntropyLoss loss;
  auto result = loss(y, ypred);

  ASSERT_LT((*result)[0], 0.01f);
}

TEST(CudaLossTest, CrossEntropyUniformPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {1, 3}, {1.0, 0.0, 0.0}, Device::CUDA, false);

  auto ypred = TensorFunctions::makeSharedTensor(
    {1, 3}, {1.0f/3, 1.0f/3, 1.0f/3}, Device::CUDA, true);

  CrossEntropyLoss loss;
  auto result = loss(y, ypred);

  ASSERT_NEAR((*result)[0], std::log(3.0f), delta);
}

TEST(CudaLossTest, CrossEntropyThrowsOnDimMismatch) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 2}, {0.5, 0.5, 0.5, 0.5}, Device::CUDA, true);

  CrossEntropyLoss loss;
  ASSERT_THROW(loss(y, ypred), std::invalid_argument);
}

TEST(CudaLossTest, CrossEntropyBackward) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 3}, {0.7, 0.2, 0.1,
         0.1, 0.8, 0.1}, Device::CUDA, true);

  CrossEntropyLoss loss;
  auto result = loss(y, ypred);
  result->backward();

  auto grads = ypred->getGrads();
  ASSERT_NEAR((*grads)[0], -0.7143f, delta);
  ASSERT_NEAR((*grads)[1],  0.0f,    delta);
  ASSERT_NEAR((*grads)[2],  0.0f,    delta);
  ASSERT_NEAR((*grads)[3],  0.0f,    delta);
  ASSERT_NEAR((*grads)[4], -0.625f,  delta);
  ASSERT_NEAR((*grads)[5],  0.0f,    delta);
}

TEST(CudaLossTest, BceForward) {
  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0, 1.0, 1.0, 0.0}, Device::CUDA, false);

  auto ypred = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.1, 0.9, 0.8, 0.2}, Device::CUDA, true);

  BceLoss loss;
  auto result = loss(y, ypred);

  const ftype expected = -(std::log(0.9f) + std::log(0.9f) +
               std::log(0.8f) + std::log(0.8f)) / 4.0f;
  ASSERT_NEAR((*result)[0], expected, delta);
}

TEST(CudaLossTest, BcePerfectPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 1}, {1.0, 0.0}, Device::CUDA, false);

  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 1}, {0.999, 0.001}, Device::CUDA, true);

  BceLoss loss;
  auto result = loss(y, ypred);

  ASSERT_LT((*result)[0], 0.01f);
}

TEST(CudaLossTest, BceRandomPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 1}, {1.0, 0.0}, Device::CUDA, false);

  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 1}, {0.5, 0.5}, Device::CUDA, true);

  BceLoss loss;
  auto result = loss(y, ypred);

  ASSERT_NEAR((*result)[0], std::log(2.0f), delta);
}

TEST(CudaLossTest, BceThrowsOnDimMismatch) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 1}, {1.0, 0.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {3, 1}, {0.5, 0.5, 0.5}, Device::CUDA, true);

  BceLoss loss;
  ASSERT_THROW(loss(y, ypred), std::invalid_argument);
}

TEST(CudaLossTest, BceNoInfOrNanOnNearZeroPred) {
  auto y = TensorFunctions::makeSharedTensor(
    {1, 1}, {1.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {1, 1}, {0.0}, Device::CUDA, true);

  BceLoss loss;
  auto result = loss(y, ypred);

  ASSERT_FALSE(std::isinf((*result)[0]));
}

TEST(CudaLossTest, BceBackward) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 1}, {1.0, 0.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {2, 1}, {0.8, 0.3}, Device::CUDA, true);

  BceLoss loss;
  auto result = loss(y, ypred);
  result->backward();

  auto grads = ypred->getGrads();
  ASSERT_NEAR((*grads)[0], -0.625f,  delta);
  ASSERT_NEAR((*grads)[1],  0.7143f, delta);
}

TEST(CudaLossTest, RmseForward) {
  auto y = TensorFunctions::makeSharedTensor(
    {3}, {1.0, 2.0, 3.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {3}, {1.5, 2.5, 2.5}, Device::CUDA, true);

  RmseLoss loss;
  auto result = loss(y, ypred);

  ASSERT_NEAR((*result)[0], 0.5f, delta);
}

TEST(CudaLossTest, RmsePerfectPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {3}, {1.0, 2.0, 3.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {3}, {1.0, 2.0, 3.0}, Device::CUDA, true);

  RmseLoss loss;
  auto result = loss(y, ypred);

  ASSERT_NEAR((*result)[0], 0.0f, delta);
}

TEST(CudaLossTest, RmseBackward) {
  auto y = TensorFunctions::makeSharedTensor(
    {2}, {1.0, 0.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {2}, {0.5, 0.5}, Device::CUDA, true);

  RmseLoss loss;
  auto result = loss(y, ypred);
  result->backward();

  auto grads = ypred->getGrads();
  ASSERT_NEAR((*grads)[0], -0.5f, delta);
  ASSERT_NEAR((*grads)[1],  0.5f, delta);
}
