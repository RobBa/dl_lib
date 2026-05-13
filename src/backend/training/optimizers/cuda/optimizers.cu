/**
 * @file optimizers.cu
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

#include "optimizers.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  // TODO: sgdStep kernel

  // TODO: rmspropStep kernel
}

namespace cuda_impl {
  void sgdStep(Tensor& param, const Tensor& grad, ftype lr) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }

  void rmspropStep(Tensor& param, Tensor& movingAvg, const Tensor& grad, ftype lr, ftype decay, ftype eps) {
    const int threadsPerBlock = DeviceProperties::getThreadsPerBlock();
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }
}
