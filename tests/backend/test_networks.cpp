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

#include "layers/ff_layer.h"

#include "activation_functions/relu.h"
#include "activation_functions/leaky_relu.h"
#include "activation_functions/softmax.h"

#include "data_modeling/tensor_functions.h"

using namespace layers;
using namespace activation;

TEST(ActivationTest, TestRelu1) {
  auto t1 = TensorFunctions::Ones({3, 2}, false);
  auto f = ReLu();

  auto res = f(t1);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], t1[i]);
  }
}

TEST(ActivationTest, TestRelu2) {
  auto t1 = TensorFunctions::Ones({3, 2}, false) * -1;
  auto f = ReLu();

  auto res = f(t1);

  constexpr ftype zero = 0; 
  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], zero);
  }
}

TEST(ActivationTest, TestLeakyRelu1) {
  auto t1 = TensorFunctions::Ones({3, 2}, false);

  auto f = LeakyReLu(0.3);
  auto res = f(t1);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], t1[i]);
  }
}

TEST(TensorOpsTest, TestLeakyRelu2) {
  auto t1 = TensorFunctions::Ones({3, 2}, false) * -1;

  constexpr ftype eps = 0.3; 
  auto f = LeakyReLu(eps);

  auto res = f(t1);

  for(size_t i=0; i<t1.getSize(); i++){
    ASSERT_DOUBLE_EQ(res[i], eps);
  }
}

TEST(LayerTest, TestFfLayer) {
  auto t1 = TensorFunctions::Ones({3, 2}, false);
  auto layer = FfLayer({2, 1}, true, false);

  auto res = layer.forward(t1);

  ASSERT_EQ(res.getDims(), Dimension({3, 1}));
}