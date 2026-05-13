/**
 * @file activation_nodes.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-23
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "activation_nodes.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  // TODO: reluBackward kernel

  // TODO: leakyReluBackward kernel

  // TODO: sigmoidBackward kernel

  // TODO: softmaxBackward kernel
}

namespace cuda_impl {
  void reluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void leakyReluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void sigmoidBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& sigmoid) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void softmaxBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& softmax) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }
}
