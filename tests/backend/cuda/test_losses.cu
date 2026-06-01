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
#include "training/loss_functions/crossentropy_softmax_loss.h"
#include "module/activation_functions/softmax.h"

#include <cmath>

using namespace std;
using namespace train;
using namespace module;

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
  ASSERT_NEAR((*result)[0], expected, 1e-4);
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

  ASSERT_NEAR((*result)[0], std::log(3.0f), 1e-4);
}

TEST(CudaLossTest, CrossEntropyForwardLarge) {
  constexpr tensorDim_t nSamples = 500;
  constexpr tensorDim_t nClasses = 200;

  auto yCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, nClasses}, Device::CPU, true) * 0.5f);
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto ypredCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, nClasses}, Device::CPU, true) * 0.7f);
  auto ypredGpu = make_shared<Tensor>(ypredCpu->createDeepCopy());
  ypredGpu->setDevice(Device::CUDA);

  CrossEntropyLoss loss;
  auto resCpu = loss(yCpu, ypredCpu);
  auto resGpu = loss(yGpu, ypredGpu);

  EXPECT_NEAR((*resCpu)[0], (*resGpu)[0], 0.1);
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
  ASSERT_NEAR((*grads)[0], -0.7143f, 1e-4);
  ASSERT_NEAR((*grads)[1],  0.0f,    1e-4);
  ASSERT_NEAR((*grads)[2],  0.0f,    1e-4);
  ASSERT_NEAR((*grads)[3],  0.0f,    1e-4);
  ASSERT_NEAR((*grads)[4], -0.625f,  1e-4);
  ASSERT_NEAR((*grads)[5],  0.0f,    1e-4);
}

TEST(CudaLossTest, CrossEntropyBackwardLarge) {
  constexpr tensorDim_t nSamples = 500;
  constexpr tensorDim_t nClasses = 200;

  auto yCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, nClasses}) * 0.5f);
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto ypredCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, nClasses}, Device::CPU, true) * 0.7f);
  auto ypredGpu = make_shared<Tensor>(ypredCpu->createDeepCopy());
  ypredGpu->setDevice(Device::CUDA);

  CrossEntropyLoss loss;
  auto resCpu = loss(yCpu, ypredCpu);
  auto resGpu = loss(yGpu, ypredGpu);

  resCpu->backward();
  resGpu->backward();

  auto gradsCpu = ypredCpu->getGrads();
  auto gradsGpu = ypredGpu->getGrads();
  gradsGpu->setDevice(Device::CPU);

  for(int i = 0; i < ypredCpu->getSize(); i++) {
    EXPECT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4) 
      << "Failed at index " << i 
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxBackward64x10) {
  constexpr tensorDim_t nSamples = 64;
  constexpr tensorDim_t nClasses = 10;

  auto yCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, nClasses}) * (1.0f / nClasses));
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto logitsCpu = make_shared<Tensor>(TensorFunctions::Gaussian({nSamples, nClasses}, 1.0f, true));
  auto logitsGpu = make_shared<Tensor>(logitsCpu->createDeepCopy());
  logitsGpu->setDevice(Device::CUDA);

  train::CrossEntropySoftmaxLoss loss;
  auto resCpu = loss(yCpu, logitsCpu);
  auto resGpu = loss(yGpu, logitsGpu);

  resCpu->backward();
  resGpu->backward();

  auto gradsCpu = logitsCpu->getGrads();
  auto gradsGpu = logitsGpu->getGrads();
  gradsGpu->setDevice(Device::CPU);

  for(int i = 0; i < logitsCpu->getSize(); i++) {
    EXPECT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4)
      << "Failed at index " << i
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxForward) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0,
             0.0, 1.0, 0.0}, Device::CUDA, false);

  auto logits = TensorFunctions::makeSharedTensor(
    {2, 3}, {0.7, 0.2, 0.1,
             0.1, 0.8, 0.1}, Device::CUDA, true);

  train::CrossEntropySoftmaxLoss loss;
  auto result = loss(y, logits);

  const float sum0 = std::exp(0.7f) + std::exp(0.2f) + std::exp(0.1f);
  const float sum1 = std::exp(0.1f) + std::exp(0.8f) + std::exp(0.1f);
  const float expected = (std::log(sum0) - 0.7f + std::log(sum1) - 0.8f) / 2.0f;
  ASSERT_NEAR((*result)[0], expected, 1e-4);
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxPerfectPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0,
             0.0, 1.0, 0.0}, Device::CUDA, false);

  auto logits = TensorFunctions::makeSharedTensor(
    {2, 3}, {100.0, 0.0, 0.0,
               0.0, 100.0, 0.0}, Device::CUDA, true);

  train::CrossEntropySoftmaxLoss loss;
  auto result = loss(y, logits);

  ASSERT_LT((*result)[0], 0.01f);
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxUniformLogits) {
  auto y = TensorFunctions::makeSharedTensor(
    {1, 3}, {1.0, 0.0, 0.0}, Device::CUDA, false);

  auto logits = TensorFunctions::makeSharedTensor(
    {1, 3}, {0.0, 0.0, 0.0}, Device::CUDA, true);

  train::CrossEntropySoftmaxLoss loss;
  auto result = loss(y, logits);

  ASSERT_NEAR((*result)[0], std::log(3.0f), 1e-4);
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxForwardLarge) {
  constexpr tensorDim_t nSamples = 500;
  constexpr tensorDim_t nClasses = 200;

  auto yCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, nClasses}) * (1.0f / nClasses));
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto logitsCpu = make_shared<Tensor>(TensorFunctions::Gaussian({nSamples, nClasses}, 1.0f, true));
  auto logitsGpu = make_shared<Tensor>(logitsCpu->createDeepCopy());
  logitsGpu->setDevice(Device::CUDA);

  train::CrossEntropySoftmaxLoss loss;
  auto resCpu = loss(yCpu, logitsCpu);
  auto resGpu = loss(yGpu, logitsGpu);

  EXPECT_NEAR((*resCpu)[0], (*resGpu)[0], 0.1f);
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxThrowsOnDimMismatch) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, Device::CUDA, false);
  auto logits = TensorFunctions::makeSharedTensor(
    {2, 2}, {0.5, 0.5, 0.5, 0.5}, Device::CUDA, true);

  train::CrossEntropySoftmaxLoss loss;
  ASSERT_THROW(loss(y, logits), std::invalid_argument);
}

TEST(CudaLossTest, CrossEntropyWithSoftmaxBackward) {
  auto y = TensorFunctions::makeSharedTensor(
    {2, 3}, {1.0, 0.0, 0.0,
             0.0, 1.0, 0.0}, Device::CUDA, false);
  auto logits = TensorFunctions::makeSharedTensor(
    {2, 3}, {0.7, 0.2, 0.1,
             0.1, 0.8, 0.1}, Device::CUDA, true);

  train::CrossEntropySoftmaxLoss loss;
  auto result = loss(y, logits);
  result->backward();

  auto grads = logits->getGrads();

  const float sum0 = std::exp(0.7f) + std::exp(0.2f) + std::exp(0.1f);
  const float sum1 = std::exp(0.1f) + std::exp(0.8f) + std::exp(0.1f);
  ASSERT_NEAR((*grads)[0], (std::exp(0.7f)/sum0 - 1.0f) / 2.0f, 1e-4);
  ASSERT_NEAR((*grads)[1], (std::exp(0.2f)/sum0)         / 2.0f, 1e-4);
  ASSERT_NEAR((*grads)[2], (std::exp(0.1f)/sum0)         / 2.0f, 1e-4);
  ASSERT_NEAR((*grads)[3], (std::exp(0.1f)/sum1)         / 2.0f, 1e-4);
  ASSERT_NEAR((*grads)[4], (std::exp(0.8f)/sum1 - 1.0f) / 2.0f, 1e-4);
  ASSERT_NEAR((*grads)[5], (std::exp(0.1f)/sum1)         / 2.0f, 1e-4);
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
  ASSERT_NEAR((*result)[0], expected, 1e-4);
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

  ASSERT_NEAR((*result)[0], std::log(2.0f), 1e-4);
}

TEST(CudaLossTest, BceForwardLarge) {
  constexpr tensorDim_t nSamples = 10000;

  auto yCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, 1}) * 0.5f);
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto ypredCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, 1}) * 0.7f);
  ypredCpu->setRequiresGrad(true);
  auto ypredGpu = make_shared<Tensor>(ypredCpu->createDeepCopy());
  ypredGpu->setDevice(Device::CUDA);
  ypredGpu->setRequiresGrad(true);

  BceLoss loss;
  auto resCpu = loss(yCpu, ypredCpu);
  auto resGpu = loss(yGpu, ypredGpu);

  EXPECT_NEAR((*resCpu)[0], (*resGpu)[0], 1e-4);
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
  ASSERT_NEAR((*grads)[0], -0.625f,  1e-4);
  ASSERT_NEAR((*grads)[1],  0.7143f, 1e-4);
}

TEST(CudaLossTest, BceBackwardLarge) {
  constexpr tensorDim_t nSamples = 10000;

  auto yCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, 1}) * 0.5f);
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto ypredCpu = make_shared<Tensor>(TensorFunctions::Ones({nSamples, 1}, Device::CPU, true) * 0.7f);
  auto ypredGpu = make_shared<Tensor>(ypredCpu->createDeepCopy());
  ypredGpu->setDevice(Device::CUDA);

  BceLoss loss;
  auto resCpu = loss(yCpu, ypredCpu);
  auto resGpu = loss(yGpu, ypredGpu);

  resCpu->backward();
  resGpu->backward();

  auto gradsCpu = ypredCpu->getGrads();
  auto gradsGpu = ypredGpu->getGrads();
  gradsGpu->setDevice(Device::CPU);

  for(int i = 0; i < ypredCpu->getSize(); i++) {
    EXPECT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4) 
      << "Failed at index " << i 
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}

TEST(CudaLossTest, RmseForward) {
  auto y = TensorFunctions::makeSharedTensor(
    {3}, {1.0, 2.0, 3.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {3}, {1.5, 2.5, 2.5}, Device::CUDA, true);

  RmseLoss loss;
  auto result = loss(y, ypred);

  ASSERT_NEAR((*result)[0], 0.5f, 1e-4);
}

TEST(CudaLossTest, RmseForwardLarge) {
  auto yCpu = make_shared<Tensor>(TensorFunctions::Gaussian({500, 500}, 1.0f));
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto ypredCpu = make_shared<Tensor>(TensorFunctions::Gaussian({500, 500}, 1.0f));
  ypredCpu->setRequiresGrad(true);
  auto ypredGpu = make_shared<Tensor>(ypredCpu->createDeepCopy());
  ypredGpu->setDevice(Device::CUDA);
  ypredGpu->setRequiresGrad(true);

  RmseLoss loss;
  auto resCpu = loss(yCpu, ypredCpu);
  auto resGpu = loss(yGpu, ypredGpu);

  EXPECT_NEAR((*resCpu)[0], (*resGpu)[0], 0.1);
}

TEST(CudaLossTest, RmsePerfectPrediction) {
  auto y = TensorFunctions::makeSharedTensor(
    {3}, {1.0, 2.0, 3.0}, Device::CUDA, false);
  auto ypred = TensorFunctions::makeSharedTensor(
    {3}, {1.0, 2.0, 3.0}, Device::CUDA, true);

  RmseLoss loss;
  auto result = loss(y, ypred);

  ASSERT_NEAR((*result)[0], 0.0f, 1e-4);
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
  ASSERT_NEAR((*grads)[0], -0.5f, 1e-4);
  ASSERT_NEAR((*grads)[1],  0.5f, 1e-4);
}

TEST(CudaLossTest, RmseBackwardLarge) {
  auto yCpu = make_shared<Tensor>(TensorFunctions::Gaussian({500, 500}, 1.0f));
  auto yGpu = make_shared<Tensor>(yCpu->createDeepCopy());
  yGpu->setDevice(Device::CUDA);

  auto ypredCpu = make_shared<Tensor>(TensorFunctions::Gaussian({500, 500}, 1.0f, true));
  auto ypredGpu = make_shared<Tensor>(ypredCpu->createDeepCopy());
  ypredGpu->setDevice(Device::CUDA);

  RmseLoss loss;
  auto resCpu = loss(yCpu, ypredCpu);
  auto resGpu = loss(yGpu, ypredGpu);

  resCpu->backward();
  resGpu->backward();

  auto gradsCpu = ypredCpu->getGrads();
  auto gradsGpu = ypredGpu->getGrads();
  gradsGpu->setDevice(Device::CPU);

  for(int i = 0; i < ypredCpu->getSize(); i++) {
    EXPECT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4)
      << "Failed at index " << i
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}
