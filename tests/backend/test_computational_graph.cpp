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

#include "computational_graph/tensor_ops/graph_creation.h"

#include "module/activation_functions/relu.h"
#include "module/activation_functions/leaky_relu.h"

#include <stdexcept>

TEST(AutogradTest, ThrowsIfNoGradientSet) {
    auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, false);
    auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, false);

    auto loss = cgraph::add(t1, t2);
    
    EXPECT_THROW(loss->backward(), std::runtime_error);
}

TEST(AutogradTest, SimpleAddition) {
    auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, true);
    auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, true);

    auto t3 = cgraph::add(t1, t2);
    auto loss = cgraph::mul(t3, t3);
    
    loss->backward();
    
    EXPECT_NEAR(t1->getGrads()->getItem(0), 10.0, 1e-5);
    EXPECT_NEAR(t2->getGrads()->getItem(0), 10.0, 1e-5);
}

TEST(AutogradTest, ScalarMultiplication) {
    auto t1 = TensorFunctions::makeSharedTensor({1}, {2.0}, true);
    auto t2 = TensorFunctions::makeSharedTensor({1}, {3.0}, true);

    auto t3 = cgraph::mul(t1, t2);
    auto loss = cgraph::mul(t3, t3);
    
    loss->backward();
    
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem(0), 36.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem(0), 24.0);
}

TEST(AutogradTest, MatMul) {
    auto t1 = TensorFunctions::makeSharedTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    auto t2 = TensorFunctions::makeSharedTensor({3, 2}, {1, 2, 3, 4, 5, 6}, true);
    
    auto t3 = cgraph::matmul(t1, t2);

    auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, true);
    for (size_t i = 0; i < t3->getSize(); ++i) {
        loss = cgraph::add(loss, cgraph::get(t3, i));
    }
    
    loss->backward();
    
    EXPECT_TRUE(t1->hasGrads());
    EXPECT_TRUE(t2->hasGrads());

    // dL/dt1 = dloss/dt3 @ t2^t = Ones({2, 2}) @ t2^t
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({0, 0}), 3.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({0, 1}), 7.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({0, 2}), 11.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({1, 0}), 3.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({1, 1}), 7.0);
    ASSERT_DOUBLE_EQ(t1->getGrads()->getItem({1, 2}), 11.0);

    // dL/dt2 = t1^t @ dloss/dt3 = t1^t @ Ones({2, 2})
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({0, 0}), 5.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({0, 1}), 5.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({1, 0}), 7.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({1, 1}), 7.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({2, 0}), 9.0);
    ASSERT_DOUBLE_EQ(t2->getGrads()->getItem({2, 1}), 9.0);
}

TEST(AutogradTest, ChainRule) {
    auto x = TensorFunctions::makeSharedTensor({1}, {2.0}, true);
    
    auto y = cgraph::mul(x, x); // y = x^2
    auto z = cgraph::add(x, y); // z = x^2 + x
    auto loss = cgraph::mul(z, z);   // loss = (x^2 + x)^2
    
    loss->backward();
    
    // dloss/dx = 2(x^2 + x) * (2x + 1)
    // At x=2: 2(4 + 2) * (4 + 1) = 2 * 6 * 5 = 60
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(0), 60.0);
}

TEST(AutogradTest, MultiVariateChainRule) {
    auto x = TensorFunctions::makeSharedTensor({2}, {1.0, 2.0}, true);
    
    auto y = cgraph::mul(x, 3.0); // y = [3, 6]
    auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, true);
    for(int i=0; i<y->getSize(); i++){
        loss = cgraph::add(loss, cgraph::get(y, i));
    }    // loss = 9
    
    loss->backward();
    
    // dloss/dx = scalar = 3
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(0), 3.0);
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(1), 3.0);

    ASSERT_DOUBLE_EQ(y->getGrads()->getItem(0), 1.0);
    ASSERT_DOUBLE_EQ(y->getGrads()->getItem(1), 1.0);
}

TEST(AutogradTest, ReLU) {
    auto x = TensorFunctions::makeSharedTensor({3}, {-1.0, 0.0, 2.0}, true);
    auto relu = module::ReLu();

    auto y = relu(x);    // [0, 0, 2]
    auto loss = cgraph::sumTensor(y);  // loss = 2
    
    loss->backward();
    
    // Gradient: [0, 0, 1] (only where input > 0)
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(0), 0.0);
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(1), 0.0);
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(2), 1.0);
}

TEST(AutogradTest, LeakyReLU) {
    auto x = TensorFunctions::makeSharedTensor({3}, {-1.0, 0.0, 2.0}, true);

    constexpr ftype eps = 0.3;
    auto relu = module::LeakyReLu(eps);

    auto y = relu(x);    // [0, 0, 2]
    auto loss = cgraph::sumTensor(y);  // loss = 2
    
    loss->backward();
    
    // Gradient: [0, 0, 1] (only where input > 0)
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(0), eps);
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(1), eps); // by convention
    ASSERT_DOUBLE_EQ(x->getGrads()->getItem(2), 1.0);
}