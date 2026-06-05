/**
 * @file test_training.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <gtest/gtest.h>

#include "data_modeling/tensor_functions.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include "net_factories.h"

#include "training/optimizers/sgd.h"
#include "training/optimizers/rmsprop.h"

#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"
#include "training/loss_functions/bce_sigmoid_loss.h"
#include "training/loss_functions/crossentropy_softmax_loss.h"

#include "training/trainers/base_train_loop.h"

using namespace std;

TEST(AutogradTest, ThrowsIfNoGradientSet) {
  auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, false);
  auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, false);

  auto loss = cgraph::add(t1, t2);
  
  ASSERT_THROW(loss->backward(), std::runtime_error);
}

TEST(AutogradTest, ChainRule) {
  auto x = TensorFunctions::makeSharedTensor({1}, {2.0}, true);
  
  auto y = cgraph::mul(x, x); // y = x^2
  auto z = cgraph::add(x, y); // z = x^2 + x
  auto loss = cgraph::mul(z, z);   // loss = (x^2 + x)^2
  
  loss->backward();
  
  // dloss/dx = 2(x^2 + x) * (2x + 1)
  // At x=2: 2(4 + 2) * (4 + 1) = 2 * 6 * 5 = 60
  ASSERT_NEAR(x->getGrads()->get(0), 60.0, 1e-5);
}

TEST(AutogradTest, MultiVariateChainRule) {
  auto x = TensorFunctions::makeSharedTensor({2}, {1.0, 2.0}, true);
  
  auto y = cgraph::mul(x, 3.0); // y = [3, 6]
  auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, true);
  for(int i = 0; i < y->getSize(); i++){
    loss = cgraph::add(loss, cgraph::get(y, i));
  }    // loss = 9
  
  loss->backward();
  
  // dloss/dx = scalar = 3
  ASSERT_NEAR(x->getGrads()->get(0), 3.0, 1e-5);
  ASSERT_NEAR(x->getGrads()->get(1), 3.0, 1e-5);

  ASSERT_NEAR(y->getGrads()->get(0), 1.0, 1e-5);
  ASSERT_NEAR(y->getGrads()->get(1), 1.0, 1e-5);
}

TEST(OverfitTest, BceSgdOverfitsSmallDataset) {
  // XOR-like: 4 samples, 2 features, binary labels
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0,
             0.0, 1.0,
             1.0, 0.0,
             1.0, 1.0}, false);

  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0,
             1.0,
             1.0,
             0.0}, false);

  auto net = makeBinaryNet();    
  auto loss = make_shared<train::BceLoss>();
  auto optim = make_shared<train::SgdOptimizer>(
    net->parameters(), /*lr=*/0.05);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/static_cast<tensorDim_t>(4));

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  // forward one more time to get final loss
  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit binary dataset\n"
    << "Final prediction: " << *pred << "\nFinal loss: " << *finalLoss;
}

TEST(OverfitTest, BceSgdOverfitsSmallDataset_OptimizedLoss) {    
  // XOR-like: 4 samples, 2 features, binary labels
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0,
             0.0, 1.0,
             1.0, 0.0,
             1.0, 1.0}, false);

  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0,
             1.0,
             1.0,
             0.0}, false);

  auto net = makeBinaryNet2();    
  auto loss = make_shared<train::BceSigmoidLoss>();
  auto optim = make_shared<train::SgdOptimizer>(
    net->parameters(), /*lr=*/0.05);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/static_cast<tensorDim_t>(4));

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  // forward one more time to get final loss
  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  auto sigmoid = module::Sigmoid();
  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit binary dataset\n"
    << "Final prediction: " << sigmoid(*pred) << "\nFinal loss: " << *finalLoss;
}

TEST(OverfitTest, CrossEntropyRMSPropOverfitsSmallDataset) {
  // 6 samples, 2 features, 3 classes
  auto x = TensorFunctions::makeSharedTensor(
    {6, 2}, {1.0, 0.0,
             1.0, 0.1,
             0.0, 1.0,
             0.1, 1.0,
             0.5, 0.5,
             0.4, 0.6}, false);

  // one-hot encoded labels
  auto y = TensorFunctions::makeSharedTensor(
    {6, 3}, {1.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             0.0, 0.0, 1.0}, false);

  auto net = makeMulticlassNet();
  auto loss = make_shared<train::CrossEntropyLoss>();
  auto optim = make_shared<train::RmsPropOptimizer>(
    net->parameters(), /*lr=*/0.001, /*decay=*/0.95);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/6);

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit multiclass dataset"
    << "Final prediction: " << *pred << "\nFinal loss: " << *finalLoss;
}

TEST(OverfitTest, CrossEntropyRMSPropOverfitsSmallDataset_OptimizedLoss) {
  // 6 samples, 2 features, 3 classes
  auto x = TensorFunctions::makeSharedTensor(
    {6, 2}, {1.0, 0.0,
             1.0, 0.1,
             0.0, 1.0,
             0.1, 1.0,
             0.5, 0.5,
             0.4, 0.6}, false);

  // one-hot encoded labels
  auto y = TensorFunctions::makeSharedTensor(
    {6, 3}, {1.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             0.0, 0.0, 1.0}, false);

  auto net = makeMulticlassNet2();
  auto loss = make_shared<train::CrossEntropySoftmaxLoss>();
  auto optim = make_shared<train::RmsPropOptimizer>(
    net->parameters(), /*lr=*/0.001, /*decay=*/0.95);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/6);

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  auto softmax = module::Softmax();
  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit multiclass dataset"
    << "Final prediction: " << softmax(*pred) << "\nFinal loss: " << *finalLoss;
}

