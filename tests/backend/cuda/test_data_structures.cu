/**
 * @file test_data_structures.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "data_modeling/tensor.h"

using namespace std;

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