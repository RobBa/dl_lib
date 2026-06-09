/**
 * @file test_shared.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-06-09
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "shared/initializers.h"

#include <cmath>
#include <vector>

using namespace utility;

static ftype sampleMean(const std::vector<ftype>& v) {
  ftype sum = 0.0f;
  for(auto x : v) sum += x;
  return sum / static_cast<ftype>(v.size());
}

static ftype sampleStd(const std::vector<ftype>& v, ftype mean) {
  ftype var = 0.0f;
  for(auto x : v) var += (x - mean) * (x - mean);
  return std::sqrt(var / static_cast<ftype>(v.size()));
}

TEST(InitializerTest, Gaussian_MeanNearZero) {
  GaussianInitializer init(1.0f, 42u);
  constexpr tensorSize_t N = 10000;
  std::vector<ftype> buf(N);
  init.fillRange(buf.data(), N);
  EXPECT_NEAR(sampleMean(buf), 0.0f, 0.05f);
}

TEST(InitializerTest, Gaussian_StddevMatchesParam) {
  const ftype stddev = 2.0f;
  GaussianInitializer init(stddev, 42u);
  constexpr tensorSize_t N = 10000;
  std::vector<ftype> buf(N);
  init.fillRange(buf.data(), N);
  ftype mean = sampleMean(buf);
  EXPECT_NEAR(sampleStd(buf, mean), stddev, 0.1f);
}

TEST(InitializerTest, Gaussian_Reproducible) {
  GaussianInitializer a(1.0f, 99u);
  GaussianInitializer b(1.0f, 99u);
  constexpr tensorSize_t N = 200;
  std::vector<ftype> va(N), vb(N);
  a.fillRange(va.data(), N);
  b.fillRange(vb.data(), N);
  for(tensorSize_t i = 0; i < N; i++)
    ASSERT_NEAR(va[i], vb[i], 1e-6f) << "Mismatch at index " << i;
}

TEST(InitializerTest, UniformXavier_ValuesInRange) {
  constexpr tensorDim_t nIn = 256, nOut = 128;
  UniformXavierInitializer init(nIn, nOut, 42u);
  const ftype range = std::sqrt(6.0f / (nIn + nOut));
  constexpr tensorSize_t N = 10000;
  std::vector<ftype> buf(N);
  init.fillRange(buf.data(), N);
  for(tensorSize_t i = 0; i < N; i++) {
    EXPECT_GE(buf[i], -range - 1e-6f) << "Below -range at index " << i;
    EXPECT_LE(buf[i],  range + 1e-6f) << "Above +range at index " << i;
  }
}

TEST(InitializerTest, UniformXavier_ZeroMean) {
  UniformXavierInitializer init(256, 128, 42u);
  constexpr tensorSize_t N = 10000;
  std::vector<ftype> buf(N);
  init.fillRange(buf.data(), N);
  EXPECT_NEAR(sampleMean(buf), 0.0f, 0.05f);
}

TEST(InitializerTest, NormalXavier_MeanNearZero) {
  NormalXavierInitializer init(256, 128, 42u);
  constexpr tensorSize_t N = 10000;
  std::vector<ftype> buf(N);
  init.fillRange(buf.data(), N);
  EXPECT_NEAR(sampleMean(buf), 0.0f, 0.05f);
}

TEST(InitializerTest, NormalXavier_StddevMatchesSigma) {
  constexpr tensorDim_t nIn = 256, nOut = 128;
  NormalXavierInitializer init(nIn, nOut, 42u);
  const ftype sigma = std::sqrt(2.0f / (nIn + nOut));
  constexpr tensorSize_t N = 10000;
  std::vector<ftype> buf(N);
  init.fillRange(buf.data(), N);
  ftype mean = sampleMean(buf);
  EXPECT_NEAR(sampleStd(buf, mean), sigma, 0.01f);
}
