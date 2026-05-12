/**
 * @file test_data_modeling.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-16
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <gtest/gtest.h>

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"

#include <stdexcept>

#include <iostream>

using namespace std;

TEST(TensorOpsTest, TestCtor) {
  auto t = Tensor({2, 2}, {2.0, 3.0, 4.0, 5.0}, Device::CPU);

  ASSERT_EQ(t.getDims(), Dimension({2, 2}));
  ASSERT_EQ(t.getDevice(), Device::CPU);
  ASSERT_TRUE(!t.getRequiresGrad());

  EXPECT_NEAR(t.get(0, 0), 2.0, 1e-5);
  EXPECT_NEAR(t.get(0, 1), 3.0, 1e-5);
  EXPECT_NEAR(t.get(1, 0), 4.0, 1e-5);
  EXPECT_NEAR(t.get(1, 1), 5.0, 1e-5);
}

TEST(TensorOpsTest, ScalarAddWorks) {
  auto t1 = TensorFunctions::Ones({2, 2});

  auto res = t1 + 1.5;

  constexpr ftype sum = 2.5;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, ScalarMulWorks) {
  auto t1 = TensorFunctions::Ones({2, 2});

  constexpr ftype f = 2.5;
  auto res = t1 * f;
    
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), f, 1e-5);
    }
  }
}

TEST(TensorOpsTest, TensorAddWorks) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 2}) * 4;

  auto res = t1 + t2;

  constexpr ftype sum = 5.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, TensorAddCanBroadCast) {
  auto t1 = TensorFunctions::Ones({3, 2, 2});
  auto t2 = Tensor({2}, {2, 3});

  auto res = t1 + t2;

  ASSERT_EQ(res.getDims(), t1.getDims());
  
  for(auto i=0; i<res.getDims().get(0); i++) {
    for(auto j=0; j<res.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j, 0), 3.0, 1e-5);
      EXPECT_NEAR(res.get(i, j, 1), 4.0, 1e-5);
    }
  }
}

TEST(TensorOpsTest, BroadcastAdd_2D) {
    // (2,3) + (3) 
    auto t1 = Tensor({2, 3}, {1.0, 2.0, 3.0,
                              4.0, 5.0, 6.0});
    auto t2 = Tensor({3}, {10.0, 20.0, 30.0});
    auto res = t1 + t2;

    // expected: each row of t1 gets t2 added elementwise
    EXPECT_NEAR(res.get(0, 0), 11.0, 1e-5);
    EXPECT_NEAR(res.get(0, 1), 22.0, 1e-5);
    EXPECT_NEAR(res.get(0, 2), 33.0, 1e-5);
    EXPECT_NEAR(res.get(1, 0), 14.0, 1e-5);
    EXPECT_NEAR(res.get(1, 1), 25.0, 1e-5);
    EXPECT_NEAR(res.get(1, 2), 36.0, 1e-5);
}

TEST(TensorOpsTest, TensorAddBroadcastNotCommutative) {
  auto t1 = TensorFunctions::Ones({3, 2, 2});
  auto t2 = Tensor({2}, {2, 3});

  EXPECT_THROW(t2 + t1, std::invalid_argument);
}

TEST(TensorOpsTest, TensorAddThrowsOnDimMismatch) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 3}) * 4;

  EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TEST(TensorOpsTest, MatrixAddGivesCorrectResults) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 2});
    
  auto res = t1 + t2;
    
  constexpr ftype resSum = 2.0;

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), resSum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, ElementwiseMulGivesCorrectResults) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 2}) * 0.5;
    
  auto res = t1 * t2;
    
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), factor, 1e-5);
    }
  }
}

TEST(TensorOpsTest, ElementwiseMulThrowsOnDimensionMismatch) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 3}) * 0.5;
    
  EXPECT_THROW(t1 * t2, std::invalid_argument);
}

TEST(TensorOpsTest, MatMulGivesCorrectValues1) {
  auto t1 = TensorFunctions::Ones({3, 2});
  auto t2 = TensorFunctions::Ones({2, 6}) * 1.5;
    
  auto res = t1.matmul(t2);
  auto expectedDims = std::vector<tensorDim_t>{3, 6};
  ASSERT_EQ(res.getDims().toVector(), expectedDims);

  constexpr ftype resSum = 3.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), resSum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, MatMulGivesCorrectValues2) {
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

  auto res = t1.matmul(t2);

  auto expectedDims = std::vector<tensorDim_t>{2, 2};
  ASSERT_EQ(res.getDims().toVector(), expectedDims);

  constexpr ftype resSum = 3.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      EXPECT_NEAR(res.get(i, j), cmpRes.get(i, j), 1e-5);
    }
  }
}

TEST(TensorOpsTest, MatMulThrowsWhenDimensionsNotMatched) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({3, 2});

  EXPECT_THROW(t1.matmul(t2), std::runtime_error);
}

TEST(TensorOpsTest, TransposeWorksAsIntended1) {
  auto t = TensorFunctions::Gaussian({3, 2}, 1.0);

  auto transposed = t.createDeepCopy();
  transposed = transposed.transpose(-1, -2);

  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(-2));
  ASSERT_EQ(t.getDims().get(-2), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto row=0; row<t.getDims().get(-2); row++) {
    for(auto col=0; col<t.getDims().get(-1); col++) {
      EXPECT_NEAR(t.get(row, col), transposed.get(col, row), 1e-5);
    }
  }
}

/**
 * @brief Swap first two dimensions.
 */
TEST(TensorOpsTest, TransposeWorksAsIntended2) {
  auto t = TensorFunctions::Gaussian({3, 2, 5}, 1.0);
  auto transposed = t.createDeepCopy().transpose(0, 1);

  ASSERT_EQ(t.getDims().get(0), transposed.getDims().get(1));
  ASSERT_EQ(t.getDims().get(1), transposed.getDims().get(0));
  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto dim1=0; dim1<t.getDims().get(0); dim1++) {
    for(auto dim2=0; dim2<t.getDims().get(1); dim2++) {
      for(auto dim3=0; dim3<t.getDims().get(-1); dim3++) {
        // we transposed dim1 and dim3
        EXPECT_NEAR(t.get(dim1, dim2, dim3), transposed.get(dim2, dim1, dim3), 1e-5);
      }
    }
  }
}

/**
 * @brief Swap first and last dimension.
 */
TEST(TensorOpsTest, TransposeWorksAsIntended3) {
  auto t = TensorFunctions::Gaussian({3, 2, 5}, 1.0);
  auto transposed = t.createDeepCopy().transpose(0, -1);

  ASSERT_EQ(t.getDims().get(0), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(0));
  ASSERT_EQ(t.getDims().get(1), transposed.getDims().get(1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto dim1=0; dim1<t.getDims().get(0); dim1++) {
    for(auto dim2=0; dim2<t.getDims().get(1); dim2++) {
      for(auto dim3=0; dim3<t.getDims().get(-1); dim3++) {
        // we transposed dim1 and dim3
        EXPECT_NEAR(t.get(dim1, dim2, dim3), transposed.get(dim3, dim2, dim1), 1e-5);
      }
    }
  }
}
