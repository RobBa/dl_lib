/**
 * @file test_module_cuda.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-13
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "data_modeling/tensor_functions.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include "module/activation_functions/relu.h"
#include "module/activation_functions/leaky_relu.h"
#include "module/activation_functions/softmax.h"
#include "module/activation_functions/sigmoid.h"
#include "module/layers/ff_layer.h"

using namespace std;

TEST(CudaActivationTest, ReluForward) {
  auto t1 = TensorFunctions::Ones({300, 500}, Device::CUDA);
  auto f = module::ReLu();

  auto res = f(t1);
  res.setDevice(Device::CPU);
  t1.setDevice(Device::CPU);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_NEAR(res[i], t1[i], 1e-5);
  }
}

TEST(CudaActivationTest, ReluInputNegative) {
  auto t1 = TensorFunctions::Ones({300, 500}, Device::CUDA) * -1;
  auto f = module::ReLu();

  auto res = f(t1);
  res.setDevice(Device::CPU);

  constexpr ftype zero = 0;
  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], zero);
  }
}

TEST(CudaAutogradTest, ReLUBackward) {
  auto x = TensorFunctions::makeSharedTensor({3}, {-1.0, 0.0, 2.0}, Device::CUDA, true);
  auto relu = module::ReLu();

  auto y = relu(x);
  auto loss = cgraph::sumTensor(y);

  loss->backward();

  ASSERT_DOUBLE_EQ(x->getGrads()->get(0), 0.0);
  ASSERT_DOUBLE_EQ(x->getGrads()->get(1), 0.0);
  ASSERT_NEAR(x->getGrads()->get(2), 1.0, 1e-5);
}

TEST(CudaActivationTest, LeakyReluForward) {
  auto t1 = TensorFunctions::Ones({300, 500}, Device::CUDA);

  auto f = module::LeakyReLu(0.3);
  auto res = f(t1);

  res.setDevice(Device::CPU);
  t1.setDevice(Device::CPU);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_NEAR(res[i], t1[i], 1e-5);
  }
}

TEST(CudaActivationTest, LeakyReluInputNegative) {
  constexpr ftype factor = -1;
  auto t1 = TensorFunctions::Ones({300, 500}, Device::CUDA) * factor;

  constexpr ftype eps = 0.3;
  auto f = module::LeakyReLu(eps);
  auto res = f(t1);

  res.setDevice(Device::CPU);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_NEAR(res[i], factor * eps, 1e-5);
  }
}

TEST(CudaAutogradTest, LeakyReLUBackward) {
  auto x = TensorFunctions::makeSharedTensor({3}, {-1.0, 0.0, 2.0}, Device::CUDA, true);

  constexpr ftype eps = 0.3;
  auto relu = module::LeakyReLu(eps);

  auto y = relu(x);
  auto loss = cgraph::sumTensor(y);

  loss->backward();

  ASSERT_DOUBLE_EQ(x->getGrads()->get(0), eps);
  ASSERT_DOUBLE_EQ(x->getGrads()->get(1), eps);
  ASSERT_NEAR(x->getGrads()->get(2), 1.0, 1e-5);
}

TEST(CudaActivationTest, SigmoidLargePositive) {
  auto t = TensorFunctions::makeSharedTensor({1}, vector<ftype>{100.0}, Device::CUDA);

  module::Sigmoid sig;
  auto res = sig(*t);

  ASSERT_NEAR(res[0], 1.0, 1e-5);
  ASSERT_FALSE(std::isnan(res[0]));
  ASSERT_FALSE(std::isinf(res[0]));
}

TEST(CudaActivationTest, SigmoidLargeNegative) {
  auto t = TensorFunctions::makeSharedTensor({1}, vector<ftype>{-100.0}, Device::CUDA);

  module::Sigmoid sig;
  auto res = sig(*t);

  ASSERT_NEAR(res[0], 0.0, 1e-5);
  ASSERT_FALSE(std::isnan(res[0]));
  ASSERT_FALSE(std::isinf(res[0]));
}

TEST(CudaAutogradTest, SigmoidBackward) {
  auto t = TensorFunctions::makeSharedTensor({2}, {0.0, 1.0}, Device::CUDA, true);

  module::Sigmoid sig;
  auto res = sig(t);
  res->backward();

  auto grads = t->getGrads();
  ASSERT_NEAR((*grads)[0], 0.25, 1e-4);
  ASSERT_NEAR((*grads)[1], 0.1966, 1e-4);
}

TEST(CudaActivationTest, SoftmaxForward) {
  auto t = TensorFunctions::makeSharedTensor({1, 3}, {1.0, 2.0, 3.0}, Device::CUDA);

  module::Softmax sm;
  auto res = sm(*t);

  ASSERT_NEAR(res[0], 0.0900, 1e-4);
  ASSERT_NEAR(res[1], 0.2447, 1e-4);
  ASSERT_NEAR(res[2], 0.6652, 1e-4);
}

TEST(CudaActivationTest, SoftmaxSumsToOne) {
  auto t = TensorFunctions::makeSharedTensor({2, 4},
                  {1.0, 2.0, 3.0, 4.0,
                   2.0, 1.0, 4.0, 3.0}, Device::CUDA);

  module::Softmax sm;
  auto res = sm(*t);

  ftype row0sum = res[0] + res[1] + res[2] + res[3];
  ftype row1sum = res[4] + res[5] + res[6] + res[7];
  ASSERT_NEAR(row0sum, 1.0, 1e-5);
  ASSERT_NEAR(row1sum, 1.0, 1e-5);
}

TEST(CudaActivationTest, SoftmaxForwardNumericalStability) {
  auto t = TensorFunctions::makeSharedTensor({1, 3}, {100.0, 101.0, 102.0}, Device::CUDA);

  module::Softmax sm;
  auto res = sm(*t);

  for(int i = 0; i < 3; i++) {
    ASSERT_FALSE(std::isnan(res[i]));
    ASSERT_FALSE(std::isinf(res[i]));
  }

  ftype rowsum = res[0] + res[1] + res[2];
  ASSERT_NEAR(rowsum, 1.0, 1e-5);
}

TEST(CudaActivationTest, SoftmaxMediumLargeInput) {
  constexpr tensorDim_t testDim = 190;
  assert(testDim <= 256 && testDim > 64); // see the kernel call 

  auto t = TensorFunctions::Gaussian({5, 10, testDim}, 2.0f, Device::CUDA);
  auto tCopy = t.createDeepCopy();
  tCopy.setDevice(Device::CPU);

  module::Softmax sm;
  auto resGpu = sm(t);
  auto resCpu = sm(tCopy);

  resGpu.setDevice(Device::CUDA);
  for(int i = 0; i < resGpu.getDims().get(0); i++) {
    for(int j = 0; j < resGpu.getDims().get(1); j++) {
      for(int k = 0; k < resGpu.getDims().get(2); k++) {
        ASSERT_NEAR(resCpu.get(i, j, k), resGpu.get(i, j, k), 1e-4)
          << "Mismatch at (" << i << ", " << j << ", " << k <<  ")"
          << " cpu=" << resCpu.get(i, j, k) 
          << " gpu=" << resGpu.get(i, j, k);
      }
    }
  }
}

TEST(CudaActivationTest, SoftmaxLargeInput) {
  constexpr tensorDim_t testDim = 1500;
  assert(testDim > 256); // see the kernel call 

  auto t = TensorFunctions::Gaussian({2, 2, testDim}, 2.0f, Device::CUDA);
  auto tCopy = t.createDeepCopy();
  tCopy.setDevice(Device::CPU);

  module::Softmax sm;
  auto resGpu = sm(t);
  auto resCpu = sm(tCopy);

  resGpu.setDevice(Device::CUDA);
  for(int i = 0; i < resGpu.getDims().get(0); i++) {
    for(int j = 0; j < resGpu.getDims().get(1); j++) {
      for(int k = 0; k < resGpu.getDims().get(2); k++) {
        ASSERT_NEAR(resCpu.get(i, j, k), resGpu.get(i, j, k), 1e-4)
          << "Mismatch at (" << i << ", " << j << ", " << k <<  ")"
          << " cpu=" << resCpu.get(i, j, k) 
          << " gpu=" << resGpu.get(i, j, k);
      }
    }
  }
}

TEST(CudaAutogradTest, SoftmaxBackward) {
  auto t = TensorFunctions::makeSharedTensor({1, 3}, {1.0, 2.0, 3.0}, Device::CUDA, true);

  module::Softmax sm;
  auto resPtr = sm(t);

  auto upstream = TensorFunctions::makeSharedTensor({1, 3}, {1.0, 0.0, 0.0}, Device::CUDA);
  resPtr->setGrads(upstream);
  resPtr->backward();

  auto grads = t->getGrads();
  ASSERT_NEAR((*grads)[0],  0.0819, 1e-5);
  ASSERT_NEAR((*grads)[1], -0.0220, 1e-5);
  ASSERT_NEAR((*grads)[2], -0.0599, 1e-5);
}

/*
TEST(CudaLayerTest, TestFfLayer) {
  auto t1 = TensorFunctions::Ones({3, 2}, Device::CUDA);
  auto layer = module::FfLayer(2, 1, Device::CUDA);

  auto res = layer(t1);

  ASSERT_EQ(res.getDims(), Dimension({3, 1}));
}
 */