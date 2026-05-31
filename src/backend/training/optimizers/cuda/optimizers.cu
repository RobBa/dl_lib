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
#include "shared/cuda/common_kernels.cuh"

#include "utility/global_params.h"

using namespace std;

namespace {
  using namespace cuda_impl;
  
  __global__ void stepSgdKernel(ftype* const params, const ftype* const grads, const ftype lr, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    params[gid] = params[gid] - lr * grads[gid];
  }

  __global__ void stepRmsPropKernel(ftype* const tensor, ftype* const v, const ftype* const grads, const ftype lr, const ftype decay, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const ftype g = grads[gid];
    const ftype mavg = decay * v[gid] + (1 - decay) * g * g;
    v[gid] = mavg;

    const ftype update = tensor[gid] - (lr * g / (cudaSqrt<ftype>(mavg) + EPS_RMSPROP));
    tensor[gid] = update;
  }
}

namespace cuda_impl {
  void sgdStep(Tensor& param, const Tensor& grads, ftype lr) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    stepSgdKernel<<<blocks, threadsPerBlock>>>(param.getData(), grads.getData(), lr, param.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void rmspropStep(Tensor& param, Tensor& movingAvg, const Tensor& grads, ftype lr, ftype decay) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    stepRmsPropKernel<<<blocks, threadsPerBlock>>>(param.getData(), movingAvg.getData(), grads.getData(), lr, decay, param.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }
}
