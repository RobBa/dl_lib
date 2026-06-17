/**
 * @file bm_matmul.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief CUDA matmul benchmarks
 * @date 2026-06-17
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif

#include <benchmark/benchmark.h>

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"

#include <cuda_runtime.h>

// BM_MatMul_CUDA(M, K, N): benchmarks left[M×K] * right[K×N] on GPU.
// Tensors are allocated on device once before the loop; only the kernel
// dispatch + synchronize is timed.
static void BM_MatMul_CUDA(benchmark::State &state)
{
  const int M = state.range(0);
  const int K = state.range(1);
  const int N = state.range(2);

  auto left = TensorFunctions::Gaussian({static_cast<tensorDim_t>(M), static_cast<tensorDim_t>(K)}, 1.0, Device::CUDA);
  auto right = TensorFunctions::Gaussian({static_cast<tensorDim_t>(K), static_cast<tensorDim_t>(N)}, 1.0, Device::CUDA);

  // warm-up: one un-timed call so driver/JIT overhead is excluded
  {
    auto res = left.matmul(right);
    cudaDeviceSynchronize();
  }

  for (auto _ : state)
  {
    auto res = left.matmul(right);
    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(res);
  }

  state.counters["GFLOP/s"] = benchmark::Counter(
      state.iterations() * 2.0 * M * K * N / 1e9,
      benchmark::Counter::kIsRate);
}

// Square sizes
BENCHMARK(BM_MatMul_CUDA)->Args({64, 64, 64})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CUDA)->Args({256, 256, 256})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CUDA)->Args({512, 512, 512})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CUDA)->Args({1024, 1024, 1024})->Unit(benchmark::kMicrosecond);

// MNIST feed-forward shapes (batch=64)
BENCHMARK(BM_MatMul_CUDA)->Args({64, 784, 256})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CUDA)->Args({64, 256, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CUDA)->Args({64, 128, 10})->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
