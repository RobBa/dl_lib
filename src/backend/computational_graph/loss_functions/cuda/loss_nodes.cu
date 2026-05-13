/**
 * @file loss_nodes.cu
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

#include "loss_nodes.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  // TODO: bceBackward kernel

  // TODO: bceSigmoidBackward kernel

  // TODO: crossEntropyBackward kernel

  // TODO: crossEntropySoftmaxBackward kernel

  // TODO: rmseBackward kernel
}

namespace cuda_impl {
  void bceBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void bceSigmoidBackward(Tensor& res, const Tensor& logits, const Tensor& yTrue) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (logits.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropyBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropySoftmaxBackward(Tensor& res, const Tensor& logits, const Tensor& yTrue) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (logits.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void rmseBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue, ftype rmse) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }
}
