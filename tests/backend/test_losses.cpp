/**
 * @file test_losses.cpp
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

#include "training/loss_functions/rmse_loss.h"
#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include <cmath>

using namespace train;

static constexpr ftype kTol = 1e-4f;

// ─── CrossEntropy ────────────────────────────────────────────────────────────

TEST(LossTest, CrossEntropyFoward) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 3}, {0.7, 0.2, 0.1,
                 0.1, 0.8, 0.1}, true);

    CrossEntropyLoss loss;
    auto result = loss(y, ypred);

    // expected: -( log(0.7) + log(0.8) ) / 2 = 0.2899
    const ftype expected = -(std::log(0.7f) + std::log(0.8f)) / 2.0f;
    EXPECT_NEAR((*result)[0], expected, kTol);
}

TEST(LossTest, CrossEntropyPerfectPrediction) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0}, false);

    // near-perfect predictions — can't use exactly 1.0 due to log(0)
    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 3}, {0.999, 0.0005, 0.0005,
                 0.0005, 0.999, 0.0005}, true);

    CrossEntropyLoss loss;
    auto result = loss(y, ypred);

    // loss should be very small
    EXPECT_LT((*result)[0], 0.01f);
}

TEST(LossTest, CrossEntropyUniformPrediction) {
    // uniform prediction should give log(3) ~ 1.0986
    auto y = TensorFunctions::makeSharedTensor(
        {1, 3}, {1.0, 0.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {1, 3}, {1.0f/3, 1.0f/3, 1.0f/3}, true);

    CrossEntropyLoss loss;
    auto result = loss(y, ypred);

    EXPECT_NEAR((*result)[0], std::log(3.0f), kTol);
}

TEST(LossTest, CrossEntropyThrowsOnDimMismatch) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 2}, {0.5, 0.5, 0.5, 0.5}, true);

    CrossEntropyLoss loss;
    EXPECT_THROW(loss(y, ypred), std::invalid_argument);
}

TEST(LossTest, CrossEntropyBackward) {
    // y = [[1,0,0],[0,1,0]], ypred = [[0.7,0.2,0.1],[0.1,0.8,0.1]]
    // grad CE w.r.t. ypred[b,i] = -y[b,i] / (ypred[b,i] * n)
    // grad[0,0] = -1/(0.7*2) = -0.7143
    // grad[0,1] =  0
    // grad[0,2] =  0
    // grad[1,0] =  0
    // grad[1,1] = -1/(0.8*2) = -0.625
    // grad[1,2] =  0
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 3}, {0.7, 0.2, 0.1,
                 0.1, 0.8, 0.1}, true);

    CrossEntropyLoss loss;
    auto result = loss(y, ypred);
    std::cout << "before bw" << std::endl;
    result->backward();
    std::cout << "past bw" << std::endl;

    auto grads = ypred->getGrads();
    EXPECT_NEAR((*grads)[0], -0.7143f, kTol);
    EXPECT_NEAR((*grads)[1],  0.0f,    kTol);
    EXPECT_NEAR((*grads)[2],  0.0f,    kTol);
    EXPECT_NEAR((*grads)[3],  0.0f,    kTol);
    EXPECT_NEAR((*grads)[4], -0.625f,  kTol);
    EXPECT_NEAR((*grads)[5],  0.0f,    kTol);
}

// ─── BCE ─────────────────────────────────────────────────────────────────────

TEST(LossTest, BceForward) {
    auto y = TensorFunctions::makeSharedTensor(
        {4, 1}, {0.0, 1.0, 1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {4, 1}, {0.1, 0.9, 0.8, 0.2}, true);

    BceLoss loss;
    auto result = loss(y, ypred);

    // expected: -( log(0.9) + log(0.9) + log(0.8) + log(0.8) ) / 4 = 0.1643
    const ftype expected = -(std::log(0.9f) + std::log(0.9f) + 
                              std::log(0.8f) + std::log(0.8f)) / 4.0f;
    EXPECT_NEAR((*result)[0], expected, kTol);
}

TEST(LossTest, BcePerfectPrediction) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 1}, {0.999, 0.001}, true);

    BceLoss loss;
    auto result = loss(y, ypred);

    EXPECT_LT((*result)[0], 0.01f);
}

TEST(LossTest, BceRandomPrediction) {
    // ypred = 0.5 for all -> loss = log(2) ~ 0.6931
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 1}, {0.5, 0.5}, true);

    BceLoss loss;
    auto result = loss(y, ypred);

    EXPECT_NEAR((*result)[0], std::log(2.0f), kTol);
}

TEST(LossTest, BceThrowsOnDimMismatch) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {3, 1}, {0.5, 0.5, 0.5}, true);

    BceLoss loss;
    EXPECT_THROW(loss(y, ypred), std::invalid_argument);
}

TEST(LossTest, BceNoInfOrNanOnNearZeroPred) {
    auto y = TensorFunctions::makeSharedTensor(
        {1, 1}, {1.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {1, 1}, {0.0}, true);

    BceLoss loss;
    auto result = loss(y, ypred);

    // clipping prevents log(0)
    EXPECT_FALSE(std::isinf((*result)[0]));
}

TEST(LossTest, BceBackward) {
    // y = [1, 0], ypred = [0.8, 0.3]
    // grad BCE w.r.t. ypred_i = (-y/ypred + (1-y)/(1-ypred)) / n
    // grad[0] = (-1/0.8 + 0) / 2 = -0.625
    // grad[1] = (0 + 1/0.7)  / 2 =  0.7143
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 1}, {0.8, 0.3}, true);

    BceLoss loss;
    auto result = loss(y, ypred);
    result->backward();

    auto grads = ypred->getGrads();
    EXPECT_NEAR((*grads)[0], -0.625f,  kTol);
    EXPECT_NEAR((*grads)[1],  0.7143f, kTol);
}

TEST(LossTest, RmseForward) {
    // y = [1, 2, 3], ypred = [1.5, 2.5, 2.5]
    // diffs = [-0.5, -0.5, 0.5]
    // MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
    // RMSE = 0.5
    auto y = TensorFunctions::makeSharedTensor(
        {3}, {1.0, 2.0, 3.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {3}, {1.5, 2.5, 2.5}, true);

    auto loss = RmseLoss{};
    auto result = loss(y, ypred);

    EXPECT_NEAR((*result)[0], 0.5f, kTol);
}

TEST(LossTest, RmsePerfectPrediction) {
    auto y = TensorFunctions::makeSharedTensor(
        {3}, {1.0, 2.0, 3.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {3}, {1.0, 2.0, 3.0}, true);

    RmseLoss loss;
    auto result = loss(y, ypred);

    EXPECT_NEAR((*result)[0], 0.0f, kTol);
}

TEST(LossTest, RmseBackward) {
    // y = [1, 0], ypred = [0.5, 0.5]
    // diffs = [0.5, -0.5], MSE = 0.25, RMSE = 0.5
    // grad_i = -(y_i - ypred_i) / (n * RMSE)
    // grad[0] = -(1 - 0.5) / (2 * 0.5) = -0.5
    // grad[1] = -(0 - 0.5) / (2 * 0.5) =  0.5
    auto y = TensorFunctions::makeSharedTensor(
        {2}, {1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {2}, {0.5, 0.5}, true);

    RmseLoss loss;
    auto result = loss(y, ypred);
    result->backward();

    auto grads = ypred->getGrads();
    EXPECT_NEAR((*grads)[0], -0.5f, kTol);
    EXPECT_NEAR((*grads)[1],  0.5f, kTol);
}