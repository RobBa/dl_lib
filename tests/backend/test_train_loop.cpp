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

    net->append(make_shared<module::FfLayer>(
        vector<tensorDim_t>{2, 4}, true, true));

    net->append(make_shared<module::LeakyReLu>(1e-5));

    net->append(make_shared<module::FfLayer>(
        vector<tensorDim_t>{4, 1}, true, true));

    net->append(make_shared<module::Sigmoid>());
    return net;
}

static shared_ptr<module::Sequential> makeMulticlassNet() {
    auto net = make_shared<module::Sequential>();

    net->append(make_shared<module::FfLayer>(
        vector<tensorDim_t>{2, 8}, true, true));

    net->append(make_shared<module::ReLu>());

    net->append(make_shared<module::FfLayer>(
        vector<tensorDim_t>{8, 3}, true, true));

    net->append(make_shared<module::Softmax>());
    return net;
}

// ─── binary overfit ─────────────────────────────────────────────────────────

TEST(OverfitTest, BCE_SGD_OverfitsSmallDataset) {
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
    cout << "Network: " << *net << endl;

    auto loss = make_shared<train::BceLoss>();
    auto optim = make_shared<train::SgdOptimizer>(
        net->parameters(), /*lr=*/0.01);

    auto trainLoop = train::BaseTrainLoop(
        net, loss, optim, /*epochs=*/1, /*bsize=*/static_cast<tensorDim_t>(4));

    trainLoop.run(x, y, /*shuffle=*/false);

    // forward one more time to get final loss
    auto pred = (*net)(x);
    auto finalLoss = (*loss)(y, pred);

    cout << "pred: " << *pred << "\nloss: " << *finalLoss << endl;

    EXPECT_LT((*finalLoss)[0], 0.05f)
        << "Network failed to overfit binary dataset";
}


// ─── multiclass overfit ──────────────────────────────────────────────────────

// TEST(OverfitTest, CrossEntropy_RMSProp_OverfitsSmallDataset) {
//     // 6 samples, 2 features, 3 classes
//     auto x = TensorFunctions::makeSharedTensor(
//         {6, 2}, {1.0, 0.0,
//                  1.0, 0.1,
//                  0.0, 1.0,
//                  0.1, 1.0,
//                  0.5, 0.5,
//                  0.4, 0.6}, false);
// 
//     // one-hot encoded labels
//     auto y = TensorFunctions::makeSharedTensor(
//         {6, 3}, {1.0, 0.0, 0.0,
//                  1.0, 0.0, 0.0,
//                  0.0, 1.0, 0.0,
//                  0.0, 1.0, 0.0,
//                  0.0, 0.0, 1.0,
//                  0.0, 0.0, 1.0}, false);
// 
//     auto net = makeMulticlassNet();
//     auto loss = make_shared<train::CrossEntropyLoss>();
//     auto optim = make_shared<train::RmsPropOptimizer>(
//         net->parameters(), /*lr=*/0.001, /*decay=*/0.9);
// 
//     auto trainLoop = train::BaseTrainLoop(
//         net, loss, optim, /*epochs=*/2000, /*bsize=*/6);
// 
//     trainLoop.run(x, y, /*shuffle=*/false);
// 
//     auto pred = (*net)(x);
//     auto finalLoss = (*loss)(y, pred);
// 
//     EXPECT_LT((*finalLoss)[0], 0.05f)
//         << "Network failed to overfit multiclass dataset";
// }