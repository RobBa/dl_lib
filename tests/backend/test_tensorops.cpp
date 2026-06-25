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

#include "computational_graph/tensor_ops/graph_creation.h"

using namespace std;

TEST(TensorOpsTest, ScalarAdd) {
  auto t1 = TensorFunctions::Ones({2, 2});

  auto res = t1 + 1.5;

  constexpr ftype sum = 2.5;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, ScalarMul) {
  auto t1 = TensorFunctions::Ones({2, 2});

  constexpr ftype f = 2.5;
  auto res = t1 * f;
    
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), f, 1e-5);
    }
  }
}

TEST(AutogradTest, ScalarMul) {
  auto t1 = TensorFunctions::makeSharedTensor({1}, {2.0}, true);
  auto t2 = TensorFunctions::makeSharedTensor({1}, {3.0}, true);

  auto t3 = cgraph::mul(t1, t2);
  auto loss = cgraph::mul(t3, t3);

  loss->backward();

  ASSERT_DOUBLE_EQ(t1->getGrads()->get(0), 36.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get(0), 24.0);
}

TEST(TensorOpsTest, TensorAdd) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 2}) * 4;

  auto res = t1 + t2;

  constexpr ftype sum = 5.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), sum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, TensorAddThrowsOnDimMismatch) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 3}) * 4;

  ASSERT_THROW(t1 + t2, std::invalid_argument);
}

TEST(AutogradTest, TensorAdd) {
  auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, true);
  auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, true);

  auto t3 = cgraph::add(t1, t2);
  auto loss = cgraph::mul(t3, t3);

  loss->backward();

  ASSERT_NEAR(t1->getGrads()->get(0), 10.0, 1e-5);
  ASSERT_NEAR(t2->getGrads()->get(0), 10.0, 1e-5);
}

TEST(TensorOpsTest, BroadcastAdd) {
  auto t1 = TensorFunctions::Ones({3, 2, 2});
  auto t2 = Tensor({2}, {2, 3});

  auto res = t1 + t2;

  ASSERT_EQ(res.getDims(), t1.getDims());
  
  for(auto i=0; i<res.getDims().get(0); i++) {
    for(auto j=0; j<res.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j, 0), 3.0, 1e-5);
      ASSERT_NEAR(res.get(i, j, 1), 4.0, 1e-5);
    }
  }
}

TEST(TensorOpsTest, BroadcastAdd2D) {
  // (2,3) + (3)
  auto t1 = Tensor({2, 3}, {1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0});
  auto t2 = Tensor({3}, {10.0, 20.0, 30.0});
  auto res = t1 + t2;

  // expected: each row of t1 gets t2 added elementwise
  ASSERT_NEAR(res.get(0, 0), 11.0, 1e-5);
  ASSERT_NEAR(res.get(0, 1), 22.0, 1e-5);
  ASSERT_NEAR(res.get(0, 2), 33.0, 1e-5);
  ASSERT_NEAR(res.get(1, 0), 14.0, 1e-5);
  ASSERT_NEAR(res.get(1, 1), 25.0, 1e-5);
  ASSERT_NEAR(res.get(1, 2), 36.0, 1e-5);
}

TEST(TensorOpsTest, BroadcastAddNotCommutative) {
  auto t1 = TensorFunctions::Ones({3, 2, 2});
  auto t2 = Tensor({2}, {2, 3});

  ASSERT_THROW(t2 + t1, std::invalid_argument);
}

TEST(AutogradTest, BroadcastAdd) {
  // gradient of broadcast add w.r.t. bias should be sum over batch dimension
  // upstream grad: (2,3) of ones → bias grad should be (3) of twos
  auto t1 = TensorFunctions::makeSharedTensor({2, 3},
    {1.0, 2.0, 3.0,
     4.0, 5.0, 6.0}, true);
  auto bias = TensorFunctions::makeSharedTensor({3},
    {0.0, 0.0, 0.0}, true);

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

TEST(TensorOpsTest, MatrixAdd) {
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 2});
    
  auto res = t1 + t2;
    
  constexpr ftype resSum = 2.0;

  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), resSum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, ElementwiseMul) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 2}) * 0.5;
    
  auto res = t1 * t2;
    
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), factor, 1e-5);
    }
  }
}

TEST(TensorOpsTest, ElementwiseMulThrowsOnDimensionMismatch) {
  constexpr ftype factor = 0.5;
  auto t1 = TensorFunctions::Ones({2, 2});
  auto t2 = TensorFunctions::Ones({2, 3}) * 0.5;
    
  ASSERT_THROW(t1 * t2, std::invalid_argument);
}

TEST(TensorOpsTest, MatMul) {
  auto t1 = TensorFunctions::Ones({3, 2});
  auto t2 = TensorFunctions::Ones({2, 6}) * 1.5;
    
  auto res = t1.matmul(t2);
  auto expectedDims = std::vector<tensorDim_t>{3, 6};
  ASSERT_EQ(res.getDims().toVector(), expectedDims);

  constexpr ftype resSum = 3.0;
  for(auto i=0; i<t1.getDims().get(0); i++) {
    for(auto j=0; j<t1.getDims().get(1); j++) {
      ASSERT_NEAR(res.get(i, j), resSum, 1e-5);
    }
  }
}

TEST(TensorOpsTest, MatMul2) {
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
      ASSERT_NEAR(res.get(i, j), cmpRes.get(i, j), 1e-5);
    }
  }
}

TEST(TensorOpsTest, MatMulTransposeLeft) {
  auto fill = [](Tensor& t, ftype a, ftype b, ftype c, ftype d) {
    t.set(a, {0,0}); t.set(b, {0,1});
    t.set(c, {1,0}); t.set(d, {1,1});
  };
  Tensor A({2, 2}), B({2, 2}), expected({2, 2});
  fill(A, 1, 2, 3, 4);
  fill(B, 5, 6, 7, 8);
  fill(expected, 26, 30, 38, 44);

  auto res = A.matmul(B, /*transposeLeft=*/true, /*transposeRight=*/false);

  ASSERT_EQ(res.getDims().toVector(), (std::vector<tensorDim_t>{2, 2}));
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 2; j++)
      ASSERT_NEAR(res.get(i, j), expected.get(i, j), 1e-5);
}

TEST(TensorOpsTest, MatMulTransposeRight) {
  auto fill = [](Tensor& t, ftype a, ftype b, ftype c, ftype d) {
    t.set(a, {0,0}); t.set(b, {0,1});
    t.set(c, {1,0}); t.set(d, {1,1});
  };
  Tensor A({2, 2}), B({2, 2}), expected({2, 2});
  fill(A, 1, 2, 3, 4);
  fill(B, 5, 6, 7, 8);
  fill(expected, 17, 23, 39, 53);

  auto res = A.matmul(B, /*transposeLeft=*/false, /*transposeRight=*/true);

  ASSERT_EQ(res.getDims().toVector(), (std::vector<tensorDim_t>{2, 2}));
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 2; j++)
      ASSERT_NEAR(res.get(i, j), expected.get(i, j), 1e-5);
}

TEST(TensorOpsTest, MatMulTransposeBoth) {
  auto fill = [](Tensor& t, ftype a, ftype b, ftype c, ftype d) {
    t.set(a, {0,0}); t.set(b, {0,1});
    t.set(c, {1,0}); t.set(d, {1,1});
  };
  Tensor A({2, 2}), B({2, 2}), expected({2, 2});
  fill(A, 1, 2, 3, 4);
  fill(B, 5, 6, 7, 8);
  fill(expected, 23, 31, 34, 46);

  auto res = A.matmul(B, /*transposeLeft=*/true, /*transposeRight=*/true);

  ASSERT_EQ(res.getDims().toVector(), (std::vector<tensorDim_t>{2, 2}));
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 2; j++)
      ASSERT_NEAR(res.get(i, j), expected.get(i, j), 1e-5);
}

TEST(AutogradTest, MatMul) {
  auto t1 = TensorFunctions::makeSharedTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
  auto t2 = TensorFunctions::makeSharedTensor({3, 2}, {1, 2, 3, 4, 5, 6}, true);

  auto t3 = cgraph::matmul(t1, t2);

  auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, true);
  for(size_t i = 0; i < t3->getSize(); ++i) {
    loss = cgraph::add(loss, cgraph::get(t3, i));
  }

  loss->backward();

  EXPECT_TRUE(t1->hasGrads());
  EXPECT_TRUE(t2->hasGrads());

  // dL/dt1 = dloss/dt3 @ t2^t = Ones({2, 2}) @ t2^t
  ASSERT_DOUBLE_EQ(t1->getGrads()->get({0, 0}), 3.0);
  ASSERT_DOUBLE_EQ(t1->getGrads()->get({0, 1}), 7.0);
  ASSERT_DOUBLE_EQ(t1->getGrads()->get({0, 2}), 11.0);
  ASSERT_DOUBLE_EQ(t1->getGrads()->get({1, 0}), 3.0);
  ASSERT_DOUBLE_EQ(t1->getGrads()->get({1, 1}), 7.0);
  ASSERT_DOUBLE_EQ(t1->getGrads()->get({1, 2}), 11.0);

  // dL/dt2 = t1^t @ dloss/dt3 = t1^t @ Ones({2, 2})
  ASSERT_DOUBLE_EQ(t2->getGrads()->get({0, 0}), 5.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get({0, 1}), 5.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get({1, 0}), 7.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get({1, 1}), 7.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get({2, 0}), 9.0);
  ASSERT_DOUBLE_EQ(t2->getGrads()->get({2, 1}), 9.0);
}


TEST(TensorOpsTest, MatrixTranspose) {
  auto t = TensorFunctions::Gaussian({3, 2}, 1.0);

  auto transposed = t.createDeepCopy();
  transposed = transposed.transpose(-1, -2);

  ASSERT_EQ(t.getDims().get(-1), transposed.getDims().get(-2));
  ASSERT_EQ(t.getDims().get(-2), transposed.getDims().get(-1));
  ASSERT_EQ(t.getDims().nDims(), transposed.getDims().nDims());
  
  for(auto row=0; row<t.getDims().get(-2); row++) {
    for(auto col=0; col<t.getDims().get(-1); col++) {
      ASSERT_NEAR(t.get(row, col), transposed.get(col, row), 1e-5);
    }
  }
}

/**
 * @brief Swap first two dimensions.
 */
TEST(TensorOpsTest, MatrixTranspose2) {
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
        ASSERT_NEAR(t.get(dim1, dim2, dim3), transposed.get(dim2, dim1, dim3), 1e-5);
      }
    }
  }
}

/**
 * @brief Swap first and last dimension.
 */
TEST(TensorOpsTest, MatrixTranspose3) {
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
        ASSERT_NEAR(t.get(dim1, dim2, dim3), transposed.get(dim3, dim2, dim1), 1e-5);
      }
    }
  }
}

TEST(TensorOpsTest, SliceRange_CorrectRowsAndDims) {
  // shape [5, 3], rows are multiples of 10 for easy identification
  auto t = Tensor({5, 3}, {
     1.0f,  2.0f,  3.0f,
     4.0f,  5.0f,  6.0f,
     7.0f,  8.0f,  9.0f,
    10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f,
  });

  auto s = t.getSlice(1, 4); // rows 1,2,3

  ASSERT_EQ(s.getDims(), Dimension({3, 3}));
  ASSERT_NEAR(s.get(0, 0),  4.0f, 1e-5f);
  ASSERT_NEAR(s.get(0, 2),  6.0f, 1e-5f);
  ASSERT_NEAR(s.get(1, 1),  8.0f, 1e-5f);
  ASSERT_NEAR(s.get(2, 0), 10.0f, 1e-5f);
  ASSERT_NEAR(s.get(2, 2), 12.0f, 1e-5f);
}

TEST(TensorOpsTest, SliceIndices_CorrectRowsAndOrder) {
  auto t = Tensor({5, 3}, {
     1.0f,  2.0f,  3.0f,
     4.0f,  5.0f,  6.0f,
     7.0f,  8.0f,  9.0f,
    10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f,
  });

  std::vector<tensorDim_t> idx = {4, 0, 2}; // reverse-pick three rows
  auto s = t.getSlice(std::span<const tensorDim_t>(idx));

  ASSERT_EQ(s.getDims(), Dimension({3, 3}));
  // row 0 of result = original row 4
  ASSERT_NEAR(s.get(0, 0), 13.0f, 1e-5f);
  ASSERT_NEAR(s.get(0, 2), 15.0f, 1e-5f);
  // row 1 of result = original row 0
  ASSERT_NEAR(s.get(1, 0),  1.0f, 1e-5f);
  ASSERT_NEAR(s.get(1, 2),  3.0f, 1e-5f);
  // row 2 of result = original row 2
  ASSERT_NEAR(s.get(2, 0),  7.0f, 1e-5f);
  ASSERT_NEAR(s.get(2, 2),  9.0f, 1e-5f);
}
