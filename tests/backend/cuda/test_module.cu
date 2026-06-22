/**
 * @file test_module.cu
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

  constexpr ftype zero = 0.0f;
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

  ASSERT_NEAR(x->getGrads()->get(0), 0.0, 1e-5);
  ASSERT_NEAR(x->getGrads()->get(1), 0.0, 1e-5);
  ASSERT_NEAR(x->getGrads()->get(2), 1.0, 1e-5);
}

TEST(CudaActivationTest, LeakyReluForward) {
  auto t1 = TensorFunctions::Ones({300, 500}, Device::CUDA);

  auto f = module::LeakyReLu(0.3);
  auto res = f(t1);

  res.setDevice(Device::CPU);
  t1.setDevice(Device::CPU);

  for(size_t i = 0; i < t1.getSize(); i++){
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

  ASSERT_NEAR(x->getGrads()->get(0), eps, 1e-5);
  ASSERT_NEAR(x->getGrads()->get(1), eps, 1e-5);
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

TEST(CudaActivationTest, SoftmaxForward64x10) {
  constexpr tensorDim_t nSamples = 64;
  constexpr tensorDim_t nClasses = 10;
  assert(nClasses <= 64); // warp-level kernel path

  auto t = TensorFunctions::Gaussian({nSamples, nClasses}, 1.0f, Device::CUDA);
  auto tCopy = t.createDeepCopy();
  tCopy.setDevice(Device::CPU);

  module::Softmax sm;
  auto resGpu = sm(t);
  auto resCpu = sm(tCopy);

  resGpu.setDevice(Device::CUDA);
  for(int i = 0; i < nSamples; i++) {
    for(int j = 0; j < nClasses; j++) {
      EXPECT_NEAR(resCpu.get(i, j), resGpu.get(i, j), 1e-4)
        << "Mismatch at (" << i << ", " << j << ")"
        << " cpu=" << resCpu.get(i, j)
        << " gpu=" << resGpu.get(i, j);
    }
  }
}

TEST(CudaActivationTest, SoftmaxMediumLarge) {
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

TEST(CudaActivationTest, SoftmaxLarge) {
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
  ASSERT_NEAR((*grads)[0],  0.0819, 1e-4);
  ASSERT_NEAR((*grads)[1], -0.0220, 1e-4);
  ASSERT_NEAR((*grads)[2], -0.0599, 1e-4);
}

TEST(CudaAutogradTest, SoftmaxBackwardBatched) {
  // 4 samples x 3 classes — exercises the one-block path with multiple strides
  auto tCpu = make_shared<Tensor>(TensorFunctions::Gaussian({4, 3}, 1.0f, true));
  auto tGpu = make_shared<Tensor>(tCpu->createDeepCopy());
  tGpu->setDevice(Device::CUDA);

  module::Softmax sm;
  auto resCpu = sm(tCpu);
  auto resGpu = sm(tGpu);

  resCpu->backward();
  resGpu->backward();

  auto gradsCpu = tCpu->getGrads();
  auto gradsGpu = tGpu->getGrads();

  for(int i = 0; i < tCpu->getSize(); i++) {
    EXPECT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4) 
      << "Failed at index " << i 
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}

TEST(CudaLossTest, SoftmaxBackward64x10) {
  // stride=10 exercises the warp-level forward and one-block backward paths
  auto tCpu = make_shared<Tensor>(TensorFunctions::Gaussian({64, 10}, 1.0f, true));
  auto tGpu = make_shared<Tensor>(tCpu->createDeepCopy());
  tGpu->setDevice(Device::CUDA);

  module::Softmax sm;
  auto resCpu = sm(tCpu);
  auto resGpu = sm(tGpu);

  resCpu->backward();
  resGpu->backward();

  auto gradsCpu = tCpu->getGrads();
  auto gradsGpu = tGpu->getGrads();
  gradsGpu->setDevice(Device::CPU);

  for(int i = 0; i < tCpu->getSize(); i++) {
    ASSERT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4)
      << "Failed at index " << i
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}

TEST(CudaAutogradTest, SoftmaxBackwardLarge) {
  constexpr tensorDim_t testDim = 1300;
  auto tCpu = make_shared<Tensor>(TensorFunctions::Gaussian({2, 2, testDim}, 2.0f, true));
  auto tGpu = make_shared<Tensor>(tCpu->createDeepCopy());
  tGpu->setDevice(Device::CUDA);

  module::Softmax sm;
  auto resPtrCpu = sm(tCpu);
  auto resPtrGpu = sm(tGpu);

  resPtrCpu->backward();
  resPtrGpu->backward();

  auto gradsCpu = tCpu->getGrads();
  auto gradsGpu = tGpu->getGrads();

  for(int i = 0; i < tCpu->getSize(); i++) {
    EXPECT_NEAR((*gradsCpu)[i], (*gradsGpu)[i], 1e-4) 
      << "Failed at index " << i 
      << " - GradsCpu[i]: " << (*gradsCpu)[i]
      << " - GradsGpu[i]: " << (*gradsGpu)[i];
  }
}

TEST(CudaLayerTest, TestFfLayer) {
  auto t1 = TensorFunctions::Ones({3, 2}, Device::CUDA);
  auto layer = module::FfLayer(2, 1, Device::CUDA);

  auto res = layer(t1);

  ASSERT_EQ(res.getDims(), Dimension({3, 1}));
}

TEST(CudaLayerTest, TestFfLayerLarge) {
  constexpr tensorDim_t largeDim = 200;
  auto tCpu = TensorFunctions::Ones({30, largeDim});

  auto tGpu = tCpu.createDeepCopy();
  tGpu.setDevice(Device::CUDA);

  auto layer = module::FfLayer(largeDim, 10);
  auto resCpu = layer(tCpu);

  layer.setDevice(Device::CUDA);
  auto resGpu = layer(tGpu);

  for(int i = 0; i < resCpu.getSize(); i++) {
    EXPECT_NEAR(resCpu[i], resGpu[i], 1e-4);
  }
}

TEST(CudaAutogradTest, FfLayerBackward) {
  auto x = TensorFunctions::makeSharedTensor({2, 3}, {
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0
  }, Device::CUDA, true);

  auto layer = module::FfLayer(3, 2, Device::CUDA, false, true);
  auto w = layer.getWeights();
  for(tensorSize_t i = 0; i < w->getSize(); i++) w->set(1.0, i);

  auto res = layer(x);
  auto loss = cgraph::sumTensor(res);
  loss->backward();

  auto xGrads = x->getGrads();
  ASSERT_NE(xGrads, nullptr);
  ASSERT_EQ(xGrads->getDims(), x->getDims());
  for(tensorSize_t i = 0; i < xGrads->getSize(); i++) {
    ASSERT_NEAR((*xGrads)[i], 2.0, 1e-5);
  }

  auto wGrads = layer.getWeights()->getGrads();
  ASSERT_NE(wGrads, nullptr);
  ASSERT_EQ(wGrads->getDims(), layer.getWeights()->getDims());
  for(tensorSize_t i = 0; i < wGrads->getSize(); i++) {
    ASSERT_NEAR((*wGrads)[i], 2.0, 1e-5);
  }
}

TEST(CudaAutogradTest, FfLayerBackwardWithBias) {
  auto x = TensorFunctions::makeSharedTensor({2, 3}, {
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0
  }, Device::CUDA, true);

  auto layer = module::FfLayer(3, 2, Device::CUDA, true, true);
  auto w = layer.getWeights();
  for(tensorSize_t i = 0; i < w->getSize(); i++) w->set(1.0, i);

  auto res = layer(x);
  auto loss = cgraph::sumTensor(res);
  loss->backward();

  auto xGrads = x->getGrads();
  ASSERT_NE(xGrads, nullptr);
  for(tensorSize_t i = 0; i < xGrads->getSize(); i++) {
    ASSERT_NEAR((*xGrads)[i], 2.0, 1e-5);
  }

  auto wGrads = layer.getWeights()->getGrads();
  ASSERT_NE(wGrads, nullptr);
  for(tensorSize_t i = 0; i < wGrads->getSize(); i++) {
    ASSERT_NEAR((*wGrads)[i], 2.0, 1e-5);
  }

  auto bGrads = layer.getBias()->getGrads();
  ASSERT_NE(bGrads, nullptr);
  ASSERT_NEAR((*bGrads)[0], 2.0, 1e-5);
  ASSERT_NEAR((*bGrads)[1], 2.0, 1e-5);
}

TEST(CudaAutogradTest, FfLayerBackwardLarge) {
  constexpr tensorDim_t inDim = 200;
  constexpr tensorDim_t outDim = 10;

  auto xCpu = make_shared<Tensor>(TensorFunctions::Gaussian({30, inDim}, 1.0f, true));
  auto xGpu = make_shared<Tensor>(xCpu->createDeepCopy());
  xGpu->setDevice(Device::CUDA);

  auto layerCpu = module::FfLayer(inDim, outDim, Device::CPU, false, true);
  auto layerGpu = module::FfLayer(inDim, outDim, Device::CUDA, false, true);

  // give both layers identical weights
  auto wCpu = layerCpu.getWeights();
  auto wGpu = layerGpu.getWeights();
  for(tensorSize_t i = 0; i < wCpu->getSize(); i++) {
    const ftype val = 1.0f / inDim;
    wCpu->set(val, i);
    wGpu->set(val, i);
  }

  auto resCpu = layerCpu(xCpu);
  auto resGpu = layerGpu(xGpu);

  resCpu->backward();
  resGpu->backward();

  auto xGradsCpu = xCpu->getGrads();
  auto xGradsGpu = xGpu->getGrads();
  for(int i = 0; i < xCpu->getSize(); i++) {
    EXPECT_NEAR((*xGradsCpu)[i], (*xGradsGpu)[i], 1e-4);
  }

  auto wGradsCpu = layerCpu.getWeights()->getGrads();
  auto wGradsGpu = layerGpu.getWeights()->getGrads();
  for(int i = 0; i < wCpu->getSize(); i++) {
    EXPECT_NEAR((*wGradsCpu)[i], (*wGradsGpu)[i], 1e-4);
  }
}

TEST(CudaAutogradTest, FfLayerBackwardWithBiasLarge) {
  constexpr tensorDim_t inDim = 200;
  constexpr tensorDim_t outDim = 10;

  auto xCpu = make_shared<Tensor>(TensorFunctions::Gaussian({30, inDim}, 1.0f, true));
  auto xGpu = make_shared<Tensor>(xCpu->createDeepCopy());
  xGpu->setDevice(Device::CUDA);

  auto layerCpu = module::FfLayer(inDim, outDim, Device::CPU, true, true);
  auto layerGpu = module::FfLayer(inDim, outDim, Device::CUDA, true, true);

  // give both layers identical weights and biases
  auto wCpu = layerCpu.getWeights();
  auto wGpu = layerGpu.getWeights();
  for(tensorSize_t i = 0; i < wCpu->getSize(); i++) {
    const ftype val = 1.0f / inDim;
    wCpu->set(val, i);
    wGpu->set(val, i);
  }

  auto bCpu = layerCpu.getBias();
  auto bGpu = layerGpu.getBias();
  for(tensorSize_t i = 0; i < bCpu->getSize(); i++) {
    bCpu->set(0.1f, i);
    bGpu->set(0.1f, i);
  }

  auto resCpu = layerCpu(xCpu);
  auto resGpu = layerGpu(xGpu);

  resCpu->backward();
  resGpu->backward();

  auto xGradsCpu = xCpu->getGrads();
  auto xGradsGpu = xGpu->getGrads();
  for(int i = 0; i < xCpu->getSize(); i++) {
    EXPECT_NEAR((*xGradsCpu)[i], (*xGradsGpu)[i], 1e-4);
  }

  auto wGradsCpu = layerCpu.getWeights()->getGrads();
  auto wGradsGpu = layerGpu.getWeights()->getGrads();
  for(int i = 0; i < wCpu->getSize(); i++) {
    EXPECT_NEAR((*wGradsCpu)[i], (*wGradsGpu)[i], 1e-4);
  }

  auto bGradsCpu = layerCpu.getBias()->getGrads();
  auto bGradsGpu = layerGpu.getBias()->getGrads();
  for(int i = 0; i < bCpu->getSize(); i++) {
    EXPECT_NEAR((*bGradsCpu)[i], (*bGradsGpu)[i], 1e-4);
  }
}