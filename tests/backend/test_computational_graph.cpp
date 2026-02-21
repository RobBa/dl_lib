/**
 * @file test_computational_graph.cpp
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

#include "computational_graph/graph_creation.h"

TEST(AutogradTest, SimpleAddition) {
    auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, true);
    auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, true);

    auto res = graph::add(t1, t2);
    auto loss = graph::mul(res, res);
    
    loss->backward();
    
    EXPECT_NEAR(t1->getGrads()->getItem(0), 10.0f, 1e-5);
    EXPECT_NEAR(t2->getGrads()->getItem(0), 10.0f, 1e-5);
}

TEST(AutogradTest, ScalarMultiplication) {
    auto t1 = TensorFunctions::makeSharedTensor({1}, {2.0}, true);
    auto t2 = TensorFunctions::makeSharedTensor({1}, {3.0}, true);

    auto res = graph::mul(t1, t2);
    auto loss = graph::mul(res, res);
    
    loss->backward();
    
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem(0), 36.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem(0), 24.0);
}

TEST(AutogradTest, MatMul) {
    auto t1 = TensorFunctions::makeSharedTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    auto t2 = TensorFunctions::makeSharedTensor({3, 2}, {1, 2, 3, 4, 5, 6}, true);
    
    auto res = graph::matmul(t1, t2);

    auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, true);
    for (size_t i = 0; i < res->getSize(); ++i) {
        loss = graph::add(loss, graph::get(res, i));
    }
    
    loss->backward();
    
    EXPECT_TRUE(t1->hasGrads());
    EXPECT_TRUE(t2->hasGrads());

    // dL/dt1 = dloss/dres @ t2^t = Ones({2, 2}) @ t2^t
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({0, 0}), 3.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({0, 1}), 7.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({0, 2}), 11.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({1, 0}), 3.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({1, 1}), 7.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({1, 2}), 11.0);

    // dL/dt2 = t1^t @ dloss/dres = t1^t @ Ones({2, 2})
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({0, 0}), 5.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({0, 1}), 5.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({1, 0}), 7.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({1, 1}), 7.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({2, 0}), 9.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({2, 1}), 9.0);
}

TEST(AutogradTest, ChainRule) {
    auto x = TensorFunctions::makeSharedTensor({1}, {2.0}, true);
    
    auto y = graph::mul(x, x); // y = x^2
    auto z = graph::add(x, y); // z = x^2 + x
    auto loss = graph::mul(z, z);   // loss = (x^2 + x)^2
    
    loss->backward();
    
    // dloss/dx = 2(x^2 + x) * (2x + 1)
    // At x=2: 2(4 + 2) * (4 + 1) = 2 * 6 * 5 = 60
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(0), 60.0);
}

TEST(AutogradTest, MultiVariateChainRule) {
    auto x = TensorFunctions::makeSharedTensor({2}, {1.0, 2.0f}, true);
    
    auto y = graph::mul(x, 3.0f); // y = [3, 6]
    auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, true);
    for(int i=0; i<y->getSize(); i++){
        loss = graph::add(loss, graph::get(y, i));
    }    // loss = 9
    
    loss->backward();
    
    // dloss/dx = scalar = 3
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(0), 3.0);
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(1), 3.0);

    ASSERT_DOUBLE_EQ(y->getGrads()->getItem(0), 1.0);
    ASSERT_DOUBLE_EQ(y->getGrads()->getItem(1), 1.0);
}

/* TEST(AutogradTest, ReLU) {
    Tensor x({3}, {-1.0f, 0.0f, 2.0f}, true);
    
    Tensor y = relu(x);    // [0, 0, 2]
    Tensor loss = sum(y);  // loss = 2
    
    loss.backward();
    
    // Gradient: [0, 0, 1] (only where input > 0)
    EXPECT_NEAR(t.getGrads()->getItem(0), 0.0f, 1e-5);
    EXPECT_NEAR(t.getGrads()->getItem(1), 0.0f, 1e-5);
    EXPECT_NEAR(t.getGrads()->getItem(2), 1.0f, 1e-5);
} */