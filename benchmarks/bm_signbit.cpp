/**
 * @file bm_signbit.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Compares std::signbit(x) vs x < 0.0f as a branch predicate.
 *
 * Key semantic difference: signbit(-NaN) == true, but (-NaN < 0.0f) == false.
 * For non-NaN inputs (the common case) both produce the same result.
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <numeric>
#include <vector>

// 1M floats, alternating sign: -0.5, +0.5, -1.5, +1.5, ...
// Pattern is regular enough to be cache-friendly but not trivially eliminatable.
static std::vector<float> makeInputs(int n) {
  std::vector<float> v(n);
  for(int i = 0; i < n; i++)
    v[i] = (i % 2 == 0 ? -1.f : 1.f) * (0.5f + static_cast<float>(i / 2));
  return v;
}

static const std::vector<float> kInputs = makeInputs(1 << 20);

static void BM_Signbit(benchmark::State& state) {
  float sum = 0.f;
  for(auto _ : state) {
    sum = 0.f;
    for(float x : kInputs) {
      if(std::signbit(x)) sum += x;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kInputs.size()));
}

static void BM_LessThanZero(benchmark::State& state) {
  float sum = 0.f;
  for(auto _ : state) {
    sum = 0.f;
    for(float x : kInputs) {
      if(x < 0.0f) sum += x;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kInputs.size()));
}

BENCHMARK(BM_Signbit);
BENCHMARK(BM_LessThanZero);

BENCHMARK_MAIN();
