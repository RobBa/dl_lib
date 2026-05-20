/**
 * @file layers.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-13
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "layers.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  // TODO: matMulPlusBias kernel
}

namespace cuda_impl {
  void matMulPlusBias(Tensor& res, const Tensor& input, const Tensor& weights, const Tensor& bias) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (res.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }
}
