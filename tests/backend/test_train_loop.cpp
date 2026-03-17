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

#include "module/networks/sequential.h"
#include "module/layers/ff_layer.h"

#include "module/activation_functions/sigmoid.h"
#include "module/activation_functions/relu.h"
#include "module/activation_functions/leaky_relu.h"
#include "module/activation_functions/softmax.h"

#include "training/optimizers/sgd.h"
#include "training/optimizers/rmsprop.h"

#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include "training/trainers/base_train_loop.h"

#include "data_modeling/tensor_functions.h"

using namespace std;

static shared_ptr<module::Sequential> makeBinaryNet() {
    auto net = make_shared<module::Sequential>();

    net->append(make_shared<module::FfLayer>(2, 4, true, true));

    net->append(make_shared<module::LeakyReLu>(0.01));

    net->append(make_shared<module::FfLayer>(4, 1, true, true));

    net->append(make_shared<module::Sigmoid>());
    return net;
}

static shared_ptr<module::Sequential> makeMulticlassNet() {
    auto net = make_shared<module::Sequential>();

    net->append(make_shared<module::FfLayer>(2, 8, true, true));

    net->append(make_shared<module::LeakyReLu>(0.01));

    net->append(make_shared<module::FfLayer>(8, 3, true, true));

    net->append(make_shared<module::Softmax>());
    return net;
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
        << "Network failed to overfit binary dataset";
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
        net->parameters(), /*lr=*/0.00001, /*decay=*/0.95);

    auto trainLoop = train::BaseTrainLoop(
        net, loss, optim, /*epochs=*/3000, /*bsize=*/6);

    trainLoop.run(x, y, /*shuffle=*/false);

    auto pred = (*net)(x);
    auto finalLoss = (*loss)(y, pred);

    EXPECT_LT((*finalLoss)[0], 0.05f)
        << "Network failed to overfit multiclass dataset";
}