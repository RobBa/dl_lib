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

#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include <cmath>

using namespace train;

static constexpr ftype kTol = 1e-4f;

// ─── CrossEntropy ────────────────────────────────────────────────────────────

TEST(LossTest, CrossEntropy_CorrectValue) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 3}, {0.7, 0.2, 0.1,
                 0.1, 0.8, 0.1}, true);

    auto loss = CrossEntropyLoss{};
    auto result = loss(y, ypred);

    // expected: -( log(0.7) + log(0.8) ) / 2 = 0.2899
    const ftype expected = -(std::log(0.7f) + std::log(0.8f)) / 2.0f;
    EXPECT_NEAR((*result)[0], expected, kTol);
}

TEST(LossTest, CrossEntropy_PerfectPrediction) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0}, false);

    // near-perfect predictions — can't use exactly 1.0 due to log(0)
    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 3}, {0.999, 0.0005, 0.0005,
                 0.0005, 0.999, 0.0005}, true);

    auto loss = CrossEntropyLoss{};
    auto result = loss(y, ypred);

    // loss should be very small
    EXPECT_LT((*result)[0], 0.01f);
}

TEST(LossTest, CrossEntropy_UniformPrediction) {
    // uniform prediction should give log(3) ~ 1.0986
    auto y = TensorFunctions::makeSharedTensor(
        {1, 3}, {1.0, 0.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {1, 3}, {1.0f/3, 1.0f/3, 1.0f/3}, true);

    auto loss = CrossEntropyLoss{};
    auto result = loss(y, ypred);

    EXPECT_NEAR((*result)[0], std::log(3.0f), kTol);
}

TEST(LossTest, CrossEntropy_DimMismatch_Throws) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 3}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 2}, {0.5, 0.5, 0.5, 0.5}, true);

    auto loss = CrossEntropyLoss{};
    EXPECT_THROW(loss(y, ypred), std::invalid_argument);
}

// ─── BCE ─────────────────────────────────────────────────────────────────────

TEST(LossTest, BCE_CorrectValue) {
    auto y = TensorFunctions::makeSharedTensor(
        {4, 1}, {0.0, 1.0, 1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {4, 1}, {0.1, 0.9, 0.8, 0.2}, true);

    auto loss = BceLoss{};
    auto result = loss(y, ypred);

    // expected: -( log(0.9) + log(0.9) + log(0.8) + log(0.8) ) / 4 = 0.1643
    const ftype expected = -(std::log(0.9f) + std::log(0.9f) + 
                              std::log(0.8f) + std::log(0.8f)) / 4.0f;
    EXPECT_NEAR((*result)[0], expected, kTol);
}

TEST(LossTest, BCE_PerfectPrediction) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 1}, {0.999, 0.001}, true);

    auto loss = BceLoss{};
    auto result = loss(y, ypred);

    EXPECT_LT((*result)[0], 0.01f);
}

TEST(LossTest, BCE_RandomPrediction) {
    // ypred = 0.5 for all -> loss = log(2) ~ 0.6931
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);

    auto ypred = TensorFunctions::makeSharedTensor(
        {2, 1}, {0.5, 0.5}, true);

    auto loss = BceLoss{};
    auto result = loss(y, ypred);

    EXPECT_NEAR((*result)[0], std::log(2.0f), kTol);
}

TEST(LossTest, BCE_DimMismatch_Throws) {
    auto y = TensorFunctions::makeSharedTensor(
        {2, 1}, {1.0, 0.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {3, 1}, {0.5, 0.5, 0.5}, true);

    auto loss = BceLoss{};
    EXPECT_THROW(loss(y, ypred), std::invalid_argument);
}

TEST(LossTest, BCE_NearZeroPred_NoInfOrNan) {
    auto y = TensorFunctions::makeSharedTensor(
        {1, 1}, {1.0}, false);
    auto ypred = TensorFunctions::makeSharedTensor(
        {1, 1}, {0.0}, true);

    auto loss = BceLoss{};
    auto result = loss(y, ypred);

    // clipping prevents log(0)
    EXPECT_FALSE(std::isinf((*result)[0]));
}