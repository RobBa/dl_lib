/**
 * @file test_layers.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <gtest/gtest.h>

#include "module/layers/ff_layer.h"

#include "module/activation_functions/relu.h"
#include "module/activation_functions/leaky_relu.h"
#include "module/activation_functions/softmax.h"
#include "module/activation_functions/sigmoid.h"

#include "data_modeling/tensor_functions.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include <cmath>

constexpr ftype delta = 1e-3;

TEST(ActivationTest, ReluForward) {
  auto t1 = TensorFunctions::Ones({3, 2}, false);
  auto f = module::ReLu();

  auto res = f(t1);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], t1[i]);
  }
}

TEST(ActivationTest, ReluInputNegative) {
  auto t1 = TensorFunctions::Ones({3, 2}, false) * -1;
  auto f = module::ReLu();

  auto res = f(t1);

  constexpr ftype zero = 0; 
  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], zero);
  }
}

TEST(AutogradTest, ReLUBackward) {
    auto x = TensorFunctions::makeSharedTensor({3}, {-1.0, 0.0, 2.0}, true);
    auto relu = module::ReLu();

    auto y = relu(x);    // [0, 0, 2]
    auto loss = cgraph::sumTensor(y);  // loss = 2
    
    loss->backward();
    
    // Gradient: [0, 0, 1] (only where input > 0)
    ASSERT_DOUBLE_EQ(x->getGrads()->get(0), 0.0);
    ASSERT_DOUBLE_EQ(x->getGrads()->get(1), 0.0);
    ASSERT_DOUBLE_EQ(x->getGrads()->get(2), 1.0);
}

TEST(ActivationTest, LeakyReluForward) {
  auto t1 = TensorFunctions::Ones({3, 2}, false);

  auto f = module::LeakyReLu(0.3);
  auto res = f(t1);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], t1[i]);
  }
}

TEST(ActivationTest, LeakyReluInputNegative) {
  auto t1 = TensorFunctions::Ones({3, 2}, false) * -1;
  
  constexpr ftype eps = 0.3;
  auto f = module::LeakyReLu(eps);
  auto res = f(t1);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], eps);
  }
}

TEST(AutogradTest, LeakyReLUBackward) {
    auto x = TensorFunctions::makeSharedTensor({3}, {-1.0, 0.0, 2.0}, true);

    constexpr ftype eps = 0.3;
    auto relu = module::LeakyReLu(eps);

    auto y = relu(x);    // [0, 0, 2]
    auto loss = cgraph::sumTensor(y);  // loss = 2
    
    loss->backward();
    
    // Gradient: [0, 0, 1] (only where input > 0)
    ASSERT_DOUBLE_EQ(x->getGrads()->get(0), eps);
    ASSERT_DOUBLE_EQ(x->getGrads()->get(1), eps); // by convention
    ASSERT_DOUBLE_EQ(x->getGrads()->get(2), 1.0);
}

TEST(ActivationTest, SigmoidForward) {
    // sigmoid(0) = 0.5, sigmoid(1) = 0.7311, sigmoid(-1) = 0.2689
    auto t = Tensor({3}, {0.0, 1.0, -1.0}, true);
    
    module::Sigmoid sig;
    auto res = sig(t);

    EXPECT_NEAR(res[0], 0.5, delta);
    EXPECT_NEAR(res[1], 0.7311, delta);
    EXPECT_NEAR(res[2], 0.2689, delta);
}

TEST(ActivationTest, SigmoidLargePositive) {
    // sigmoid(100) should be ~1, not inf or nan
    auto t = Tensor({1}, {100.0}, true);
    
    module::Sigmoid sig;
    auto res = sig(t);

    EXPECT_NEAR(res[0], 1.0, delta);
    EXPECT_FALSE(std::isnan(res[0]));
    EXPECT_FALSE(std::isinf(res[0]));
}

TEST(ActivationTest, SigmoidLargeNegative) {
    // sigmoid(-100) should be ~0, not nan
    auto t = Tensor({1}, {-100.0}, true);
    
    module::Sigmoid sig;
    auto res = sig(t);

    EXPECT_NEAR(res[0], 0.0, delta);
    EXPECT_FALSE(std::isnan(res[0]));
    EXPECT_FALSE(std::isinf(res[0]));
}

TEST(AutogradTest, SigmoidBackward) {
    // grad of sigmoid = sigmoid(x) * (1 - sigmoid(x))
    // for x=0: grad = 0.5 * 0.5 = 0.25
    // for x=1: grad = 0.7311 * 0.2689 = 0.1966
    auto t = TensorFunctions::makeSharedTensor(
        {2}, {0.0, 1.0}, true);
    
    module::Sigmoid sig;
    auto res = sig(t);
    res->backward();

    auto grads = t->getGrads();
    EXPECT_NEAR((*grads)[0], 0.25, delta);
    EXPECT_NEAR((*grads)[1], 0.1966, delta);
}

TEST(ActivationTest, SoftmaxForward) {
    // softmax([1, 2, 3])
    // exp([1,2,3]) = [2.7183, 7.3891, 20.0855]
    // sum = 30.1929
    // softmax = [0.0900, 0.2447, 0.6652]
    auto t = Tensor({1, 3}, {1.0, 2.0, 3.0}, true);
    
    module::Softmax sm;
    auto res = sm(t);

    EXPECT_NEAR(res[0], 0.0900, delta);
    EXPECT_NEAR(res[1], 0.2447, delta);
    EXPECT_NEAR(res[2], 0.6652, delta);
}

TEST(ActivationTest, SoftmaxSumsToOne) {
    auto t = Tensor({2, 4}, 
                    {1.0, 2.0, 3.0, 4.0,
                     2.0, 1.0, 4.0, 3.0}, 
                     true);
    
    module::Softmax sm;
    auto res = sm(t);

    // each row must sum to 1
    ftype row0sum = res[0] + res[1] + res[2] + res[3];
    ftype row1sum = res[4] + res[5] + res[6] + res[7];
    EXPECT_NEAR(row0sum, 1.0, delta);
    EXPECT_NEAR(row1sum, 1.0, delta);
}

TEST(ActivationTest, SoftmaxForwardNumericalStability) {
    // large values should not produce nan or inf
    auto t = Tensor({1, 3}, {100.0, 101.0, 102.0}, true);
    
    module::Softmax sm;
    auto res = sm(t);

    for(int i = 0; i < 3; i++) {
        EXPECT_FALSE(std::isnan(res[i]));
        EXPECT_FALSE(std::isinf(res[i]));
    }
    ftype rowsum = res[0] + res[1] + res[2];
    EXPECT_NEAR(rowsum, 1.0, delta);
}

TEST(AutogradTest, SoftmaxBackward) {
    // for softmax with upstream grad of ones, the gradient is zero
    // because d/dx_i sum(softmax(x)) = 0 (softmax sums to 1 always)
    // more useful: upstream = [1, 0, 0]
    // grad[i] = softmax[i] * (upstream[i] - dot(upstream, softmax))
    // for x=[1,2,3], softmax=[0.09, 0.2447, 0.6652]
    // dot([1,0,0], softmax) = 0.09
    // grad[0] = 0.09   * (1 - 0.09)   =  0.0819
    // grad[1] = 0.2447 * (0 - 0.09)   = -0.0220
    // grad[2] = 0.6652 * (0 - 0.09)   = -0.0599
    auto t = TensorFunctions::makeSharedTensor(
        {1, 3}, {1.0, 2.0, 3.0}, true);
    
    module::Softmax sm;
    auto resPtr = sm(t);
    
    // set upstream gradient to [1, 0, 0]
    auto upstream = TensorFunctions::makeSharedTensor(
        {1, 3}, {1.0, 0.0, 0.0}, false);
    resPtr->setGrads(upstream);
    resPtr->backward();

    auto grads = t->getGrads();
    EXPECT_NEAR((*grads)[0],  0.0819, delta);
    EXPECT_NEAR((*grads)[1], -0.0220, delta);
    EXPECT_NEAR((*grads)[2], -0.0599, delta);
}

TEST(LayerTest, TestFfLayer) {
  auto t1 = TensorFunctions::Ones({3, 2}, false);
  auto layer = module::FfLayer(2, 1, true, false);

  auto res = layer(t1);

  ASSERT_EQ(res.getDims(), Dimension({3, 1}));
}