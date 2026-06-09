/**
 * @file test_tensorops.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-05-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"

#include "computational_graph/tensor_ops/graph_creation.h"

TEST(CudaTensorOpsTest, TestCtor) {
  auto t = Tensor({2, 2}, {2.0, 3.0, 4.0, 5.0}, Device::CUDA);

  ASSERT_EQ(t.getDims(), Dimension({2, 2}));
  ASSERT_EQ(t.getDevice(), Device::CUDA);
  ASSERT_TRUE(!t.getRequiresGrad());

  ASSERT_NEAR(t.get(0, 0), 2.0, 1e-5);
  ASSERT_NEAR(t.get(0, 1), 3.0, 1e-5);
  ASSERT_NEAR(t.get(1, 0), 4.0, 1e-5);
  ASSERT_NEAR(t.get(1, 1), 5.0, 1e-5);
}

TEST(CudaTensorOpsTest, ScalarAdd) {
  auto t1 = TensorFunctions::Ones({500, 500}, Device::CUDA);

  auto res = t1 + 1.5;
  res.setDevice(Device::CPU);

  constexpr ftype sum = 2.5;
  for(auto i = 0; i < t1.getDims().get(0); i++) {
    for(auto j = 0; j < t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, ScalarMul) {
  auto t1 = TensorFunctions::Ones({500, 500}, Device::CUDA);

  constexpr ftype f = 2.5;
  auto res = t1 * f;
  res.setDevice(Device::CPU);

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), f, 1e-5);
    }
  }
}

TEST(CudaAutogradTest, ScalarMul) {
  auto t1 = TensorFunctions::makeSharedTensor({1}, {2.0}, Device::CUDA, true);
  auto t2 = TensorFunctions::makeSharedTensor({1}, {3.0}, Device::CUDA, true);

  auto t3 = cgraph::mul(t1, t2);
  auto loss = cgraph::mul(t3, t3);

  loss->backward();

  ASSERT_DOUBLE_EQ(t1->getGrads()->get(0), 36.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get(0), 24.0);
}

TEST(CudaTensorOpsTest, TensorAdd) {
  auto t1 = TensorFunctions::Ones({500, 500}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({500, 500}, Device::CUDA) * 4;

  auto res = t1 + t2;
  res.setDevice(Device::CPU);

  constexpr ftype sum = 5.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, TensorAddThrowsOnDimMismatch) {
  auto t1 = TensorFunctions::Ones({2, 2}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({2, 3}, Device::CUDA) * 4;

  ASSERT_THROW(t1 + t2, std::invalid_argument);
}

TEST(CudaAutogradTest, TensorAdd) {
  auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, Device::CUDA, true);
  auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, Device::CUDA, true);

  auto t3 = cgraph::add(t1, t2);
  auto loss = cgraph::mul(t3, t3);

  loss->backward();

  ASSERT_NEAR(t1->getGrads()->get(0), 10.0, 1e-5);
  ASSERT_NEAR(t2->getGrads()->get(0), 10.0, 1e-5);
}

TEST(CudaTensorOpsTest, BroadcastAdd) {
  auto t1 = TensorFunctions::Ones({3, 2, 2}, Device::CUDA);
  auto t2 = Tensor({2}, {2, 3}, Device::CUDA);

  auto res = t1 + t2;

  ASSERT_EQ(res.getDims(), t1.getDims());
  
  for(auto i=0; i<res.getDims().get(0); i++) {
    for(auto j=0; j<res.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j, 0), 3.0, 1e-5);
      ASSERT_NEAR(res.get(i, j, 1), 4.0, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, BroadcastAdd_2D) {
  // (2,3) + (3)
  auto t1 = Tensor({2, 3}, {1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0}, Device::CUDA);
  auto t2 = Tensor({3}, {10.0, 20.0, 30.0}, Device::CUDA);

  auto res = t1 + t2;

  // expected: each row of t1 gets t2 added elementwise
  ASSERT_NEAR(res.get(0, 0), 11.0, 1e-5);
  ASSERT_NEAR(res.get(0, 1), 22.0, 1e-5);
  ASSERT_NEAR(res.get(0, 2), 33.0, 1e-5);
  ASSERT_NEAR(res.get(1, 0), 14.0, 1e-5);
  ASSERT_NEAR(res.get(1, 1), 25.0, 1e-5);
  ASSERT_NEAR(res.get(1, 2), 36.0, 1e-5);
}

TEST(CudaTensorOpsTest, BroadcastAddNotCommutative) {
  auto t1 = TensorFunctions::Ones({3, 2, 2}, Device::CUDA);
  auto t2 = Tensor({2}, {2, 3}, Device::CUDA);

  ASSERT_THROW(t2 + t1, std::invalid_argument);
}

TEST(CudaAutogradTest, BroadcastAdd) {
  auto t1 = TensorFunctions::makeSharedTensor({2, 3},
    {1.0, 2.0, 3.0,
     4.0, 5.0, 6.0}, Device::CUDA, true);
  auto bias = TensorFunctions::makeSharedTensor({3},
    {0.0, 0.0, 0.0}, Device::CUDA, true);

  auto res = cgraph::add(t1, bias);
  res->backward();

  // bias grad should be sum over batch: [2, 2, 2]
  auto biasGrad = bias->getGrads();
  ASSERT_NEAR((*biasGrad)[0], 2.0, 1e-5);
  ASSERT_NEAR((*biasGrad)[1], 2.0, 1e-5);
  ASSERT_NEAR((*biasGrad)[2], 2.0, 1e-5);

  // t1 grad should be ones (add is identity for non-broadcast operand)
  auto t1Grad = t1->getGrads();
  for(int i = 0; i < 6; i++) {
    ASSERT_NEAR((*t1Grad)[i], 1.0, 1e-5);
  }
}

TEST(CudaAutogradTest, BroadcastAddLarge) {
  constexpr int dimsize = 1500;

  auto t1 = std::make_shared<Tensor>(
    TensorFunctions::Gaussian({2, dimsize},
      5.0, Device::CPU, true));

  auto bias = std::make_shared<Tensor>(
    TensorFunctions::Gaussian({dimsize},
      2.0, Device::CPU, true));

  auto t1Gpu = std::make_shared<Tensor>(t1->createDeepCopy());
  t1Gpu->setDevice(Device::CUDA);
  auto biasGpu = std::make_shared<Tensor>(bias->createDeepCopy());
  biasGpu->setDevice(Device::CUDA);

  auto resCpu = cgraph::add(t1, bias);
  resCpu->backward();

  auto resGpu = cgraph::add(t1Gpu, biasGpu);
  resGpu->backward();
  resGpu->setDevice(Device::CPU);

  auto biasGrad = bias->getGrads();
  auto biasGradGpu = biasGpu->getGrads();
  for(int i = 0; i < biasGrad->getSize(); i++) {
    ASSERT_NEAR((*biasGrad)[i], (*biasGradGpu)[i], 1e-5);
  }

  auto t1Grad = t1->getGrads();
  auto t1GpuGrads = t1Gpu->getGrads();
  for(int i = 0; i < t1Grad->getSize(); i++) {
    ASSERT_NEAR((*t1Grad)[i], (*t1GpuGrads)[i], 1e-5);
  }
}

TEST(CudaTensorOpsTest, MatrixAdd) {
  auto t1 = TensorFunctions::Ones({200, 200}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({200, 200}, Device::CUDA);
    
  auto res = t1 + t2;
  res.setDevice(Device::CPU);

  constexpr ftype resSum = 2.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), resSum, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, ElementwiseMul) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({200, 200}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({200, 200}, Device::CUDA) * 0.5;
    
  auto res = t1 * t2;
  res.setDevice(Device::CPU);

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), factor, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, ElementwiseMulThrowsOnDimensionMismatch) {
  auto t1 = TensorFunctions::Ones({2, 2}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({2, 3}, Device::CUDA) * 0.5;
    
  ASSERT_THROW(t1 * t2, std::invalid_argument);
}

TEST(CudaTensorOpsTest, MatMul) {
  auto t1 = Tensor({2, 2});
  auto t2 = Tensor({2, 2});

  auto cmpRes = Tensor({2, 2});

  auto populateTensor = [](Tensor& t, ftype v1, ftype v2, ftype v3, ftype v4) {
    t.set(v1, {0, 0});
    t.set(v2, {0, 1});
    t.set(v3, {1, 0});
    t.set(v4, {1, 1});
  };

  populateTensor(t1, 1, 2, 3, 4);
  populateTensor(t2, 5, 6, 7, 8);
  populateTensor(cmpRes, 19, 22, 43, 50);

  t1.setDevice(Device::CUDA);
  t2.setDevice(Device::CUDA);

  auto res = t1.matmul(t2);
  res.setDevice(Device::CPU);

  auto expectedDims = std::vector<tensorDim_t>{2, 2};
  ASSERT_EQ(res.getDims().toVector(), expectedDims);

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), cmpRes.get(i, j), 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, MatMulLarge) {
  auto t1 = TensorFunctions::Gaussian({1000, 10}, 2.0);
  auto t2 = TensorFunctions::Gaussian({10, 1000}, 2.0);
  auto resCpu = t1.matmul(t2);

  t1.setDevice(Device::CUDA);
  t2.setDevice(Device::CUDA);

  auto resGpu = t1.matmul(t2);
  resGpu.setDevice(Device::CPU);

  const auto expectedDims = resCpu.getDims().toVector();
  ASSERT_EQ(resGpu.getDims().toVector(), expectedDims);

  for(auto i = 0; i < resCpu.getDims().get(0); i++) {
    for(auto j = 0; j < resCpu.getDims().get(1); j++) {
      ASSERT_NEAR(resCpu.get(i, j), resGpu.get(i, j), 1e-4)     
        << "Mismatch at (" << i << ", " << j << ")"
        << " cpu=" << resCpu.get(i, j) 
        << " gpu=" << resGpu.get(i, j);
    }
  }
}

TEST(CudaAutogradTest, MatMul) {
  constexpr int dimsize = 30;

  // init tensors
  auto t1 = std::make_shared<Tensor>(
    TensorFunctions::Gaussian(
      {10, dimsize}, 2.0, Device::CPU, true));

  auto t2 = std::make_shared<Tensor>(
    TensorFunctions::Gaussian(
      {dimsize, 10}, 2.0, Device::CPU, true));

  auto t1Gpu = std::make_shared<Tensor>(t1->createDeepCopy());
  auto t2Gpu = std::make_shared<Tensor>(t2->createDeepCopy());
  t1Gpu->setDevice(Device::CUDA);
  t2Gpu->setDevice(Device::CUDA);

  {
    // compute and take loss
    auto resCpu = cgraph::matmul(t1, t2);
    auto lossCpu = TensorFunctions::makeSharedTensor(
      {1}, {0.0}, Device::CPU, true);
    for(size_t i = 0; i < resCpu->getSize(); ++i) {
      lossCpu = cgraph::add(lossCpu, cgraph::get(resCpu, i));
    }
    lossCpu->backward();
  }

  {
    auto resGpu = cgraph::matmul(t1Gpu, t2Gpu);
    auto lossGpu = TensorFunctions::makeSharedTensor(
      {1}, {0.0}, Device::CUDA, true);
    for(size_t i = 0; i < resGpu->getSize(); ++i) {
      lossGpu = cgraph::add(lossGpu, cgraph::get(resGpu, i));
    }
    lossGpu->backward();
  }

  // get grads and compare
  auto t1Grads = t1->getGrads();
  auto t2Grads = t2->getGrads();

  auto t1GpuGrads = t1Gpu->getGrads();
  auto t2GpuGrads = t2Gpu->getGrads();

  for(auto i = 0; i < t1Grads->getDims().get(0); i++) {
    for(auto j = 0; j < t1Grads->getDims().get(1); j++) {    
      EXPECT_NEAR(t1Grads->get(i, j), t1GpuGrads->get(i, j), 1e-5)
        << "Mismatch at (" << i << ", " << j << ")"
        << " cpu=" << t1Grads->get(i, j) 
        << " gpu=" << t1GpuGrads->get(i, j);
    }
  }

  for(auto i = 0; i < t2Grads->getDims().get(0); i++) {
    for(auto j = 0; j < t2Grads->getDims().get(1); j++) {    
      EXPECT_NEAR(t2Grads->get(i, j), t2GpuGrads->get(i, j), 1e-5)
        << "Mismatch at (" << i << ", " << j << ")"
        << " cpu=" << t2Grads->get(i, j) 
        << " gpu=" << t2GpuGrads->get(i, j);
    }
  }
}

TEST(CudaTensorOpsTest, MatrixTranspose1) {
  auto t = TensorFunctions::Gaussian({200, 200}, 1.0, Device::CUDA);

  auto transposed = t.createDeepCopy();
  transposed = transposed.transpose(-1, -2);
  transposed.setDevice(Device::CPU);

  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(-2));
  ASSERT_EQ(t.getDims().get(-2), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto row=0; row<t.getDims().get(-2); row++) {
    for(auto col=0; col<t.getDims().get(-1); col++) {
      ASSERT_NEAR(t.get(row, col), transposed.get(col, row), 1e-5)
        << "Mismatch at (" << row << ", " << col << ")"
        << " t=" << t.get(row, col) 
        << " transposed=" << transposed.get(col, row);
    }
  }
}

// Swap first two dimensions.
TEST(CudaTensorOpsTest, MatrixTranspose2) {
  auto t = TensorFunctions::Gaussian({10, 20, 200}, 1.0, Device::CUDA);
  auto transposed = t.createDeepCopy().transpose(0, 1);
  transposed.setDevice(Device::CPU);

  ASSERT_EQ(t.getDims().get(0), transposed.getDims().get(1));
  ASSERT_EQ(t.getDims().get(1), transposed.getDims().get(0));
  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto dim1=0; dim1<t.getDims().get(0); dim1++) {
    for(auto dim2=0; dim2<t.getDims().get(1); dim2++) {
      for(auto dim3=0; dim3<t.getDims().get(-1); dim3++) {
        // we transposed dim1 and dim3
        ASSERT_NEAR(t.get(dim1, dim2, dim3), transposed.get(dim2, dim1, dim3), 1e-5)
          << "Mismatch at (" << dim1 << ", " << dim2 << ", " << dim3 << ")"
          << " t=" << t.get(dim1, dim2, dim3)
          << " transposed=" << transposed.get(dim2, dim1, dim3);
      }
    }
  }
}

// Swap first and last dimension.
TEST(CudaTensorOpsTest, MatrixTranspose3) {
  auto t = TensorFunctions::Gaussian({10, 20, 200}, 1.0, Device::CUDA);
  auto transposed = t.createDeepCopy().transpose(0, -1);
  transposed.setDevice(Device::CPU);

  ASSERT_EQ(t.getDims().get(0), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(0));
  ASSERT_EQ(t.getDims().get(1), transposed.getDims().get(1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto dim1=0; dim1<t.getDims().get(0); dim1++) {
    for(auto dim2=0; dim2<t.getDims().get(1); dim2++) {
      for(auto dim3=0; dim3<t.getDims().get(-1); dim3++) {
        // we transposed dim1 and dim3
        ASSERT_NEAR(t.get(dim1, dim2, dim3), transposed.get(dim3, dim2, dim1), 1e-5)
          << "Mismatch at (" << dim1 << ", " << dim2 << ", " << dim3 << ")"
          << " t=" << t.get(dim1, dim2, dim3)
          << " transposed=" << transposed.get(dim3, dim2, dim1);
      }
    }
  }
}
TEST(CudaTensorOpsTest, SliceRange_CorrectRowsAndDims) {
  auto t = Tensor({5, 3}, {
     1.0f,  2.0f,  3.0f,
     4.0f,  5.0f,  6.0f,
     7.0f,  8.0f,  9.0f,
    10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f,
  }, Device::CUDA);

  auto s = t.getSlice(1, 4); // rows 1,2,3

  ASSERT_EQ(s.getDims(), Dimension({3, 3}));
  ASSERT_NEAR(s.get(0, 0),  4.0f, 1e-4f);
  ASSERT_NEAR(s.get(0, 2),  6.0f, 1e-4f);
  ASSERT_NEAR(s.get(1, 1),  8.0f, 1e-4f);
  ASSERT_NEAR(s.get(2, 0), 10.0f, 1e-4f);
  ASSERT_NEAR(s.get(2, 2), 12.0f, 1e-4f);
}

TEST(CudaTensorOpsTest, SliceIndices_CorrectRowsAndOrder) {
  auto t = Tensor({5, 3}, {
     1.0f,  2.0f,  3.0f,
     4.0f,  5.0f,  6.0f,
     7.0f,  8.0f,  9.0f,
    10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f,
  }, Device::CUDA);

  std::vector<tensorDim_t> idx = {4, 0, 2};
  auto s = t.getSlice(std::span<const tensorDim_t>(idx));

  ASSERT_EQ(s.getDims(), Dimension({3, 3}));
  ASSERT_NEAR(s.get(0, 0), 13.0f, 1e-4f);
  ASSERT_NEAR(s.get(0, 2), 15.0f, 1e-4f);
  ASSERT_NEAR(s.get(1, 0),  1.0f, 1e-4f);
  ASSERT_NEAR(s.get(1, 2),  3.0f, 1e-4f);
  ASSERT_NEAR(s.get(2, 0),  7.0f, 1e-4f);
  ASSERT_NEAR(s.get(2, 2),  9.0f, 1e-4f);
}
