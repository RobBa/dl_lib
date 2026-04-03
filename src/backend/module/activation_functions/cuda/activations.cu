/**
 * @file activations.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-31
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "activations.cuh"

#include "utility/cuda/cuda_common.cuh"

namespace {
  __global__ void reluKernel(ftype* res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid>=size)
      return;

    constexpr ftype zero = 0;
    res[gid] = fmaxf(input[gid], zero);
  }

  __global__ void leakyReluKernel(ftype* res, const ftype* const input, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid>=size)
      return;

    res[gid] = fmaxf(input[gid], eps*input[gid]); // eps < 1
  }

  __device__ __forceinline__ ftype sigmoid(ftype x) {
      ftype z = expf(-fabsf(x));
      ftype s = 1.0f / (1.0f + z);
      return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
  }

  __global__ void sigmoidKernel(ftype* res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid>=size)
      return;

    res[gid] = sigmoid(input[gid]);
  }
}


namespace cuda {
  void relu(Tensor& res, const Tensor& in) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

    reluKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), in.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void leakyRelu(Tensor& res, const Tensor& in, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

    leakyReluKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), eps, in.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void sigmoid(Tensor& res, const Tensor& in) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

    sigmoidKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), in.getSize());
    cudaErrchk(cudaDeviceSynchronize());
}

  void softmax(Tensor& res, const Tensor& in) {
    static_assert(false);
  }
}