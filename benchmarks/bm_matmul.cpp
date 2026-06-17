/**
 * @file bm_matmul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief CPU matmul benchmarks
 * @date 2026-06-17
 */

#include <benchmark/benchmark.h>

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"

// BM_MatMul_CPU(M, K, N): benchmarks left[M×K] * right[K×N] on CPU.
static void BM_MatMul_CPU(benchmark::State &state)
{
  const int M = state.range(0);
  const int K = state.range(1);
  const int N = state.range(2);

  auto left = TensorFunctions::Gaussian({static_cast<tensorDim_t>(M), static_cast<tensorDim_t>(K)}, 1.0);
  auto right = TensorFunctions::Gaussian({static_cast<tensorDim_t>(K), static_cast<tensorDim_t>(N)}, 1.0);

  for (auto _ : state)
  {
    auto res = left.matmul(right);
    benchmark::DoNotOptimize(res);
  }

  state.counters["GFLOP/s"] = benchmark::Counter(
      state.iterations() * 2.0 * M * K * N / 1e9,
      benchmark::Counter::kIsRate);
}

// Square sizes
BENCHMARK(BM_MatMul_CPU)->Args({64, 64, 64})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CPU)->Args({256, 256, 256})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CPU)->Args({512, 512, 512})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CPU)->Args({1024, 1024, 1024})->Unit(benchmark::kMicrosecond);

// MNIST feed-forward shapes (batch=64)
BENCHMARK(BM_MatMul_CPU)->Args({64, 784, 256})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CPU)->Args({64, 256, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatMul_CPU)->Args({64, 128, 10})->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
