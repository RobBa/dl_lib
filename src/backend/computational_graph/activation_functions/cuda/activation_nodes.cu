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
  __global__ void reluBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const parent, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;

    res[gid] =  parent[gid] > 0 ? upstreamGrad[gid] : 0;
  }

  __global__ void leakyReluBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const parent, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;

    res[gid] = parent[gid] > 0 ? upstreamGrad[gid] : eps * upstreamGrad[gid];
  }

  __global__ void sigmoidBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const sigmoid, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;

    ftype si = sigmoid[gid];
    res[gid] = si * (1 - si) * upstreamGrad[gid];
  }

  // TODO: softmaxBackward kernel
}

namespace cuda_impl {
  void reluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    reluBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), parent.getData(), res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void leakyReluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), parent.getData(), eps, res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void sigmoidBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& sigmoid) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    sigmoidBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), sigmoid.getData(), res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void softmaxBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& softmax) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel

    cudaErrchk(cudaDeviceSynchronize());
  }
}
