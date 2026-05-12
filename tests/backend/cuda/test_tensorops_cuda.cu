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

TEST(TensorOpsTest_CUDA, ScalarAddWorks) {
  auto t1 = TensorFunctions::Ones({2, 2}, Device::CUDA, false);

  auto res = t1 + 1.5;

  constexpr ftype sum = 2.5;
  for(auto i = 0; i < t1.getDims().get(0); i++) {
    for(auto j = 0; j < t1.getDims().get(1); j++) {
      ASSERT_DOUBLE_EQ(res.get(i, j), sum);
    }
  }
}