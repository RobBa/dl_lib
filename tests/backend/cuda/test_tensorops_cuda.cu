/**
 * @file test_tensorops_cuda.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-05-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <gtest/gtest.h>

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"

TEST(CudaTensorOpsTest, ScalarAddWorks) {
  auto t1 = TensorFunctions::Ones({10000, 10000}, Device::CUDA);

  auto res = t1 + 1.5;
  res.setDevice(Device::CPU);

  constexpr ftype sum = 2.5;
  for(auto i = 0; i < t1.getDims().get(0); i++) {
    for(auto j = 0; j < t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, ScalarMulWorks) {
  auto t1 = TensorFunctions::Ones({10000, 10000}, Device::CUDA);

  constexpr ftype f = 2.5;
  auto res = t1 * f;
  res.setDevice(Device::CPU);

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), f, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, TensorAddWorks) {
  auto t1 = TensorFunctions::Ones({10000, 10000}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({10000, 10000}, Device::CUDA) * 4;

  auto res = t1 + t2;
  res.setDevice(Device::CPU);

  constexpr ftype sum = 5.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(CudaTensorOpsTest, TensorAddCanBroadCast) {
  auto t1 = TensorFunctions::Ones({3, 2, 2}, Device::CUDA);
  auto t2 = Tensor({2}, {2, 3}, Device::CUDA);

  auto res = t1 + t2;

  ASSERT_EQ(res.getDims(), t1.getDims());
  
  for(auto i=0; i<res.getDims().get(0); i++) {
    for(auto j=0; j<res.getDims().get(1); j++) {
      ASSERT_DOUBLE_EQ(res.get(i, j, 0), 3.0);
      ASSERT_DOUBLE_EQ(res.get(i, j, 1), 4.0);
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
    ASSERT_DOUBLE_EQ(res.get(0, 0), 11.0);
    ASSERT_DOUBLE_EQ(res.get(0, 1), 22.0);
    ASSERT_DOUBLE_EQ(res.get(0, 2), 33.0);
    ASSERT_DOUBLE_EQ(res.get(1, 0), 14.0);
    ASSERT_DOUBLE_EQ(res.get(1, 1), 25.0);
    ASSERT_DOUBLE_EQ(res.get(1, 2), 36.0);
}

TEST(CudaTensorOpsTest, TensorAddBroadcastNotCommutative) {
  auto t1 = TensorFunctions::Ones({3, 2, 2}, Device::CUDA);
  auto t2 = Tensor({2}, {2, 3}, Device::CUDA);

  EXPECT_THROW(t2 + t1, std::invalid_argument);
}

TEST(CudaTensorOpsTest, TensorAddThrowsOnDimMismatch) {
  auto t1 = TensorFunctions::Ones({2, 2}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({2, 3}, Device::CUDA) * 4;

  EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TEST(CudaTensorOpsTest, MatrixAddGivesCorrectResults) {
  auto t1 = TensorFunctions::Ones({200, 20000}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({200, 20000}, Device::CUDA);
    
  auto res = t1 + t2;
  res.setDevice(Device::CPU);

  constexpr ftype resSum = 2.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_DOUBLE_EQ(res.get(i, j), resSum);
    }
  }
}

TEST(CudaTensorOpsTest, ElementwiseMulGivesCorrectResults) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({200, 20000}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({200, 20000}, Device::CUDA) * 0.5;
    
  auto res = t1 * t2;
  res.setDevice(Device::CPU);

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_DOUBLE_EQ(res.get(i, j), factor);
    }
  }
}

TEST(CudaTensorOpsTest, ElementwiseMulThrowsOnDimensionMismatch) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({2, 2}, Device::CUDA);
  auto t2 = TensorFunctions::Ones({2, 3}, Device::CUDA) * 0.5;
    
  EXPECT_THROW(t1 * t2, std::invalid_argument);
}