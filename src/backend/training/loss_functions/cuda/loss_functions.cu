/**
 * @file loss_functions.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-10
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "loss_functions.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  template<typename T>
  __forceinline__ __device__ T bce(T y, T ypred) {
    if constexpr (std::is_same_v<T, float>) {
      return y * __logf(max(ypred, EPS_BCE)) + (1 - y) * __logf(max(1-ypred, EPS_BCE));
    }
    else if constexpr (std::is_same_v<T, double>) {
      return y * log(max(ypred, EPS_BCE)) + (1 - y) * log(max(1-ypred, EPS_BCE));
    }
    else {
      static_assert(always_false<T>, "Unexpected value for ftype");
    }
  }

  __global__ void bceKernel(ftype* const res, const ftype* const y, const ftype* const ypred, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;

    int tid = threadIdx.x;
    extern __shared__ ftype sdata[];

    // pre-load first round
    {
      int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
      sdata[tid] = bce<ftype>(y[i], ypred[i]) + bce<ftype>(y[i + blockDim.x], ypred[i + blockDim.x]);
      __syncthreads();
    }

    for(tensorSize_t i = blockDim.x / 2; i >= 64; i >>= 1){
      if(tid < i) {
        sdata[tid] += sdata[tid + i];
      }
      __syncthreads();
    }

    if(tid < 32 && gid + 32 < size) {
      sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();

    if(tid < 16 && gid + 16 < size) {
      sdata[tid] += sdata[tid + 16];
    }
    __syncthreads();

    if(tid < 8 && gid + 8 < size) {
      sdata[tid] += sdata[tid + 8];
    }
    __syncthreads();

    if(tid < 4 && gid + 4 < size) {
      sdata[tid] += sdata[tid + 4];
    }
    __syncthreads();

    if(tid < 2 && gid + 2 < size) {
      sdata[tid] += sdata[tid + 2];
    }
    __syncthreads();

    if(tid == 0 && gid + 1 < size) {
      sdata[0] = (sdata[0] + sdata[1]) / size;
    }
  }

  // TODO: bceSigmoidLoss kernel

  // TODO: crossEntropyLoss kernel

  // TODO: crossEntropySoftmaxLoss kernel

  // TODO: rmseLoss kernel
}

namespace cuda_impl {
  Tensor bceLoss(const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / (threadsPerBlock * 2);

    Tensor res(vector<tensorDim_t>{1}, Device::CUDA, true);
    bceKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
        res.getData(), y.getData(), yPred.getData(), y.getDims()[0]);
    cudaErrchk(cudaDeviceSynchronize());

    return res;
  }

  Tensor bceSigmoidLoss(const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    Tensor res(vector<tensorDim_t>{1}, Device::CUDA, true);
    // TODO: launch kernel
    cudaErrchk(cudaDeviceSynchronize());

    return res;
  }

  Tensor crossEntropyLoss(const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    Tensor res(vector<tensorDim_t>{1}, Device::CUDA, true);
    // TODO: launch kernel
    cudaErrchk(cudaDeviceSynchronize());

    return res;
  }

  Tensor crossEntropySoftmaxLoss(const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    Tensor res(vector<tensorDim_t>{1}, Device::CUDA, true);
    // TODO: launch kernel
    cudaErrchk(cudaDeviceSynchronize());

    return res;
  }

  Tensor rmseLoss(const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    Tensor res(vector<tensorDim_t>{1}, Device::CUDA, true);
    // TODO: launch kernel
    cudaErrchk(cudaDeviceSynchronize());

    return res;
  }
}
