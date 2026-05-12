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
  auto t1 = TensorFunctions::Ones({10000, 10000}, Device::CUDA, false);

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
  auto t1 = TensorFunctions::Ones({10000, 10000}, Device::CUDA, false);

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
  auto t1 = TensorFunctions::Ones({10000, 10000}, Device::CUDA, false);
  auto t2 = TensorFunctions::Ones({10000, 10000}, Device::CUDA, false) * 4;

  auto res = t1 + t2;
  res.setDevice(Device::CPU);

  constexpr ftype sum = 5.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

