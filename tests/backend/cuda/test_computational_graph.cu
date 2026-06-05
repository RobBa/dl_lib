/**
 * @file test_computational_graph.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

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

TEST(CudaAutogradTest, ThrowsIfNoGradientSet) {
  auto t1 = TensorFunctions::makeSharedTensor({1}, {3.0}, Device::CUDA, false);
  auto t2 = TensorFunctions::makeSharedTensor({1}, {2.0}, Device::CUDA, false);

  auto loss = cgraph::add(t1, t2);

  ASSERT_THROW(loss->backward(), std::runtime_error);
}

TEST(CudaAutogradTest, ChainRule) {
  auto x = TensorFunctions::makeSharedTensor({1}, {2.0}, Device::CUDA, true);

  auto y = cgraph::mul(x, x);      // y = x^2
  auto z = cgraph::add(x, y);      // z = x^2 + x
  auto loss = cgraph::mul(z, z);   // loss = (x^2 + x)^2

  loss->backward();

  // dloss/dx = 2(x^2 + x) * (2x + 1)
  // At x=2: 2(4 + 2) * (4 + 1) = 60
  ASSERT_NEAR(x->getGrads()->get(0), 60.0, 1e-4);
}

TEST(CudaAutogradTest, MultiVariateChainRule) {
  auto x = TensorFunctions::makeSharedTensor({2}, {1.0, 2.0}, Device::CUDA, true);

  auto y = cgraph::mul(x, 3.0); // y = [3, 6]
  auto loss = TensorFunctions::makeSharedTensor({1}, {0.0}, Device::CUDA, true);
  for(int i = 0; i < y->getSize(); i++){
    loss = cgraph::add(loss, cgraph::get(y, i));
  }    // loss = 9

  loss->backward();

  // dloss/dx = 3
  ASSERT_NEAR(x->getGrads()->get(0), 3.0, 1e-4);
  ASSERT_NEAR(x->getGrads()->get(1), 3.0, 1e-4);

  ASSERT_NEAR(y->getGrads()->get(0), 1.0, 1e-4);
  ASSERT_NEAR(y->getGrads()->get(1), 1.0, 1e-4);
}

TEST(CudaOverfitTest, BceSgdOverfitsSmallDataset) {
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0,
             0.0, 1.0,
             1.0, 0.0,
             1.0, 1.0}, Device::CUDA, false);

  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0,
             1.0,
             1.0,
             0.0}, Device::CUDA, false);

  auto net = makeBinaryNet(Device::CUDA);
  auto loss = make_shared<train::BceLoss>();
  auto optim = make_shared<train::SgdOptimizer>(
    net->parameters(), /*lr=*/0.05);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/static_cast<tensorDim_t>(4));

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit binary dataset\n"
    << "Final loss: " << *finalLoss;
}

TEST(CudaOverfitTest, BceSgdOverfitsSmallDataset_OptimizedLoss) {
  auto x = TensorFunctions::makeSharedTensor(
    {4, 2}, {0.0, 0.0,
             0.0, 1.0,
             1.0, 0.0,
             1.0, 1.0}, Device::CUDA, false);

  auto y = TensorFunctions::makeSharedTensor(
    {4, 1}, {0.0,
             1.0,
             1.0,
             0.0}, Device::CUDA, false);

  auto net = makeBinaryNet2(Device::CUDA);
  auto loss = make_shared<train::BceSigmoidLoss>();
  auto optim = make_shared<train::SgdOptimizer>(
    net->parameters(), /*lr=*/0.05);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/static_cast<tensorDim_t>(4));

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit binary dataset\n"
    << "Final loss: " << *finalLoss;
}

TEST(CudaOverfitTest, CrossEntropyRMSPropOverfitsSmallDataset) {
  auto x = TensorFunctions::makeSharedTensor(
    {6, 2}, {1.0, 0.0,
             1.0, 0.1,
             0.0, 1.0,
             0.1, 1.0,
             0.5, 0.5,
             0.4, 0.6}, Device::CUDA, false);

  auto y = TensorFunctions::makeSharedTensor(
    {6, 3}, {1.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             0.0, 0.0, 1.0}, Device::CUDA, false);

  auto net = makeMulticlassNet(Device::CUDA);
  auto loss = make_shared<train::CrossEntropyLoss>();
  auto optim = make_shared<train::RmsPropOptimizer>(
    net->parameters(), /*lr=*/0.001, /*decay=*/0.95);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/6);

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);


  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit multiclass dataset\n"
    << "Final loss: " << *finalLoss;
}

TEST(CudaOverfitTest, CrossEntropyRMSPropOverfitsSmallDataset_OptimizedLoss) {
  auto x = TensorFunctions::makeSharedTensor(
    {6, 2}, {1.0, 0.0,
             1.0, 0.1,
             0.0, 1.0,
             0.1, 1.0,
             0.5, 0.5,
             0.4, 0.6}, Device::CUDA, false);

  auto y = TensorFunctions::makeSharedTensor(
    {6, 3}, {1.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             0.0, 0.0, 1.0}, Device::CUDA, false);

  auto net = makeMulticlassNet2(Device::CUDA);
  auto loss = make_shared<train::CrossEntropySoftmaxLoss>();
  auto optim = make_shared<train::RmsPropOptimizer>(
    net->parameters(), /*lr=*/0.001, /*decay=*/0.95);

  auto trainLoop = train::BaseTrainLoop(
    net, loss, optim, /*epochs=*/2000, /*bsize=*/6);

  trainLoop.run(x, y, /*shuffle=*/false, /*verbose=*/false);

  auto pred = (*net)(x);
  auto finalLoss = (*loss)(y, pred);

  EXPECT_LT((*finalLoss)[0], 0.05f)
    << "Network failed to overfit multiclass dataset\n"
    << "Final loss: " << *finalLoss;
}


