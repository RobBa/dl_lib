/**
 * @file test_shared.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-06-09
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "shared/initializers.h"

#include <cmath>
#include <vector>
#include <cuda_runtime.h>

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

TEST(CudaInitializerTest, Gaussian_GPU_MeanAndStddev) {
  const ftype stddev = 1.0f;
  GaussianInitializer init(stddev, 42u);
  constexpr tensorSize_t N = 10000; // must be even for curandGenerateNormal
  ftype* d_data;
  cudaMalloc(&d_data, N * sizeof(ftype));

  init.fillRangeGpu(d_data, N);

  std::vector<ftype> buf(N);
  cudaMemcpy(buf.data(), d_data, N * sizeof(ftype), cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  ftype mean = sampleMean(buf);
  EXPECT_NEAR(mean, 0.0f, 0.05f);
  EXPECT_NEAR(sampleStd(buf, mean), stddev, 0.1f);
}

TEST(CudaInitializerTest, UniformXavier_GPU_ValuesInRange) {
  constexpr tensorDim_t nIn = 256, nOut = 128;
  UniformXavierInitializer init(nIn, nOut, 42u);
  const ftype range = std::sqrt(6.0f / (nIn + nOut));
  constexpr tensorSize_t N = 10000;
  ftype* d_data;
  cudaMalloc(&d_data, N * sizeof(ftype));

  init.fillRangeGpu(d_data, N);

  std::vector<ftype> buf(N);
  cudaMemcpy(buf.data(), d_data, N * sizeof(ftype), cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  for(tensorSize_t i = 0; i < N; i++) {
    EXPECT_GE(buf[i], -range - 1e-5f) << "Below -range at index " << i;
    EXPECT_LE(buf[i],  range + 1e-5f) << "Above +range at index " << i;
  }
}

TEST(CudaInitializerTest, NormalXavier_GPU_MeanAndStddev) {
  constexpr tensorDim_t nIn = 256, nOut = 128;
  NormalXavierInitializer init(nIn, nOut, 42u);
  const ftype sigma = std::sqrt(2.0f / (nIn + nOut));
  constexpr tensorSize_t N = 10000; // must be even for curandGenerateNormal
  ftype* d_data;
  cudaMalloc(&d_data, N * sizeof(ftype));

  init.fillRangeGpu(d_data, N);

  std::vector<ftype> buf(N);
  cudaMemcpy(buf.data(), d_data, N * sizeof(ftype), cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  ftype mean = sampleMean(buf);
  EXPECT_NEAR(mean, 0.0f, 0.05f);
  EXPECT_NEAR(sampleStd(buf, mean), sigma, 0.01f);
}
