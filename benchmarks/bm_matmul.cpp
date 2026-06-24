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

// BM_MatMul_TransposeLeft_CPU: left[K×M]^T * right[K×N] → result[M×N]
static void BM_MatMul_TransposeLeft_CPU(benchmark::State &state)
{
  const int M = state.range(0);
  const int K = state.range(1);
  const int N = state.range(2);

  auto left  = TensorFunctions::Gaussian({static_cast<tensorDim_t>(K), static_cast<tensorDim_t>(M)}, 1.0);
  auto right = TensorFunctions::Gaussian({static_cast<tensorDim_t>(K), static_cast<tensorDim_t>(N)}, 1.0);

  for (auto _ : state)
  {
    auto res = left.matmul(right, /*transposeLeft=*/true, /*transposeRight=*/false);
    benchmark::DoNotOptimize(res);
  }

  state.counters["GFLOP/s"] = benchmark::Counter(
      state.iterations() * 2.0 * M * K * N / 1e9,
      benchmark::Counter::kIsRate);
}

// BM_MatMul_TransposeRight_CPU: left[M×K] * right[N×K]^T → result[M×N]
static void BM_MatMul_TransposeRight_CPU(benchmark::State &state)
{
  const int M = state.range(0);
  const int K = state.range(1);
  const int N = state.range(2);

  auto left  = TensorFunctions::Gaussian({static_cast<tensorDim_t>(M), static_cast<tensorDim_t>(K)}, 1.0);
  auto right = TensorFunctions::Gaussian({static_cast<tensorDim_t>(N), static_cast<tensorDim_t>(K)}, 1.0);

  for (auto _ : state)
  {
    auto res = left.matmul(right, /*transposeLeft=*/false, /*transposeRight=*/true);
    benchmark::DoNotOptimize(res);
  }

  state.counters["GFLOP/s"] = benchmark::Counter(
      state.iterations() * 2.0 * M * K * N / 1e9,
      benchmark::Counter::kIsRate);
}

// BM_MatMul_TransposeBoth_CPU: left[K×M]^T * right[N×K]^T → result[M×N]
static void BM_MatMul_TransposeBoth_CPU(benchmark::State &state)
{
  const int M = state.range(0);
  const int K = state.range(1);
  const int N = state.range(2);

  auto left  = TensorFunctions::Gaussian({static_cast<tensorDim_t>(K), static_cast<tensorDim_t>(M)}, 1.0);
  auto right = TensorFunctions::Gaussian({static_cast<tensorDim_t>(N), static_cast<tensorDim_t>(K)}, 1.0);

  for (auto _ : state)
  {
    auto res = left.matmul(right, /*transposeLeft=*/true, /*transposeRight=*/true);
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

// Transpose variants — same shape set
#define BM_TRANSPOSE_ARGS(BM) \
  BENCHMARK(BM)->Args({64,   64,   64  })->Unit(benchmark::kMicrosecond); \
  BENCHMARK(BM)->Args({256,  256,  256 })->Unit(benchmark::kMicrosecond); \
  BENCHMARK(BM)->Args({512,  512,  512 })->Unit(benchmark::kMicrosecond); \
  BENCHMARK(BM)->Args({1024, 1024, 1024})->Unit(benchmark::kMicrosecond); \
  BENCHMARK(BM)->Args({64,   784,  256 })->Unit(benchmark::kMicrosecond); \
  BENCHMARK(BM)->Args({64,   256,  128 })->Unit(benchmark::kMicrosecond); \
  BENCHMARK(BM)->Args({64,   128,  10  })->Unit(benchmark::kMicrosecond);

BM_TRANSPOSE_ARGS(BM_MatMul_TransposeLeft_CPU)
BM_TRANSPOSE_ARGS(BM_MatMul_TransposeRight_CPU)
BM_TRANSPOSE_ARGS(BM_MatMul_TransposeBoth_CPU)

#undef BM_TRANSPOSE_ARGS

BENCHMARK_MAIN();
