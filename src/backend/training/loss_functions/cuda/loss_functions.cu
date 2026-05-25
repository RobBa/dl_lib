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

#include "utility/macros.h"
#include "utility/cuda/cuda_common.cuh"

#include "shared/cuda/common_kernels.cuh"
#include "shared/cuda/common_softmax.cuh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace std;

namespace {
  using namespace cuda_impl;

  template<typename T>
  __forceinline__ __device__ T bce(T y, T ypred) {
    if constexpr (std::is_same_v<T, float>) {
      return y * __logf(cudaMax<T>(ypred, EPS_BCE)) + (1 - y) * __logf(cudaMax<T>(1 - ypred, EPS_BCE));
    }
    else if constexpr (std::is_same_v<T, double>) {
      return y * log(cudaMax<T>(ypred, EPS_BCE)) + (1 - y) * log(cudaMax<T>(1 - ypred, EPS_BCE));
    }
    else {
      static_assert(always_false<T>, "Unexpected value for ftype");
    }
  }

  /**
   * @brief Forward BCE loss.
   */
  __global__ void bceLossKernel(ftype* const res, const ftype* const y, const ftype* const ypred, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;

    int tid = threadIdx.x;
    extern __shared__ ftype smem[];

    // pre-load first round
    {
      int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
      smem[tid] = bce<ftype>(y[i], ypred[i]) + bce<ftype>(y[i + blockDim.x], ypred[i + blockDim.x]);
      __syncthreads();
    }

    for(tensorSize_t i = blockDim.x / 2; i >= 64; i >>= 1){
      if(tid < i) {
        smem[tid] += smem[tid + i];
      }
      __syncthreads();
    }

    // TODO: warp shuffles
    volatile ftype* sdata = smem;
    if(tid < 32) {
      if(gid + 32 < size) {
        sdata[tid] += sdata[tid + 32];
      }
      if(gid + 16 < size) {
        sdata[tid] += sdata[tid + 16];
      }
      if(gid + 8 < size) {
        sdata[tid] += sdata[tid + 8];
      }
      if(gid + 4 < size) {
        sdata[tid] += sdata[tid + 4];
      }
      if(gid + 2 < size) {
        sdata[tid] += sdata[tid + 2];
      }
      if(gid + 1 < size) {
        sdata[0] = (sdata[0] + sdata[1]) / size;
      }
    }

    if(tid == 0) {
      res[blockIdx.x] = sdata[0];
    }
  }

  template<typename T>
  __forceinline__ __device__ T bceSimplified(T y, T logit) {
    constexpr T zero = 0;
    if constexpr (std::is_same_v<T, float>) {
      return cudaMax<T>(logit, zero) - logit * y + __logf(1 + __expf(abs(logit)));
    }
    else if constexpr (std::is_same_v<T, double>) {
      return cudaMax<T>(logit, zero) - logit * y + log(1 + exp(abs(logit)));
    }
    else {
      static_assert(always_false<T>, "Unexpected value for ftype");
    }
  }

  /**
   * @brief BCE kernel with integrated sigmoid.
   * 
   * @param logits Forward logits.
   */
  __global__ void bceSigmoidLossKernel(ftype* const res, const ftype* const y, const ftype* const logits, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;
    
    int tid = threadIdx.x;
    extern __shared__ ftype smem[];

    // pre-load first round
    {
      int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
      smem[tid] = bceSimplified<ftype>(y[i], logits[i]) + bceSimplified<ftype>(y[i + blockDim.x], logits[i + blockDim.x]);
      __syncthreads();
    }

    for(tensorSize_t i = blockDim.x / 2; i >= 64; i >>= 1){
      if(tid < i) {
        smem[tid] += smem[tid + i];
      }
      __syncthreads();
    }

    // TODO: warp shuffles
    volatile ftype* sdata = smem;
    if(tid < 32) {
      if(gid + 32 < size) {
        sdata[tid] += sdata[tid + 32];
      }
      if(gid + 16 < size) {
        sdata[tid] += sdata[tid + 16];
      }
      if(gid + 8 < size) {
        sdata[tid] += sdata[tid + 8];
      }
      if(gid + 4 < size) {
        sdata[tid] += sdata[tid + 4];
      }
      if(gid + 2 < size) {
        sdata[tid] += sdata[tid + 2];
      }
      if(gid + 1 < size) {
        sdata[0] = (sdata[0] + sdata[1]) / size;
      }
    }

    if(tid == 0) {
      res[blockIdx.x] = sdata[0];
    }
  }

  template<typename T>
  __forceinline__ __device__ ftype crossEntropy(const ftype y, const ftype ypred) {
    if constexpr (std::is_same_v<T, float>) {
      return y * __logf(cudaMax<T>(ypred, EPS_CROSSENTROPY));
    }
    else if constexpr (std::is_same_v<T, double>) {
      return y * log(cudaMax<T>(ypred, EPS_CROSSENTROPY));
    }
    else {
      static_assert(always_false<T>, "Encountered unexpected ftype");
    }
  }

  /**
   * @brief Cross-entropy reduction over a full block. Covers two times blockDim.x
   */
  __global__ void crossEntropyLossKernelOneBlock(ftype* const res, const ftype* const y, const ftype* const yPred, const tensorSize_t size) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ ftype smem[];
    smem[tid] = crossEntropy<ftype>(y[gid], yPred[gid]);
    if(gid + blockDim.x < size) {
      smem[tid] += crossEntropy<ftype>(y[gid + blockDim.x], yPred[gid + blockDim.x]);
    }
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 64; offset >>= 2) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle again
    volatile ftype* sdata = smem;
    if(tid < 32) {
      if(gid + 32 < size) {
        sdata[tid] += sdata[tid + 32];
      }
      if(gid + 16 < size) {
        sdata[tid] += sdata[tid + 16];
      }
      if(gid + 8 < size) {
        sdata[tid] += sdata[tid + 8];
      }
      if(gid + 4 < size) {
        sdata[tid] += sdata[tid + 4];
      }
      if(gid + 2 < size) {
        sdata[tid] += sdata[tid + 2];
      }
      if(gid + 1 < size) {
        sdata[0] = (sdata[0] + sdata[1]) / size;
      }
    }

    if(threadIdx.x == 0) {
      res[blockDim.x] = sdata[0];
    }
  }

  // TODO: crossEntropySoftmaxLoss kernel

  /**
   * @brief Helper for RMSE loss.
   */
  __forceinline__ __device__ ftype diffPow(const ftype y, const ftype ypred) {
    auto diff = y - ypred;
    return diff * diff;
  }

  /**
   * @brief RMSE forward loss.
   */
  __global__ void rmseKernelOneBlock(ftype* const res, const ftype* const y, const ftype* const yPred, const tensorSize_t size) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ ftype smem[];
    smem[tid] = diffPow(y[gid], yPred[gid]);
    if(gid + blockDim.x < size) {
      smem[tid] += diffPow(y[gid + blockDim.x], yPred[gid + blockDim.x]);
    }
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 64; offset >>= 2) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle again
    volatile ftype* sdata = smem;
    if(tid < 32) {
      if(gid + 32 < size) {
        sdata[tid] += sdata[tid + 32];
      }
      if(gid + 16 < size) {
        sdata[tid] += sdata[tid + 16];
      }
      if(gid + 8 < size) {
        sdata[tid] += sdata[tid + 8];
      }
      if(gid + 4 < size) {
        sdata[tid] += sdata[tid + 4];
      }
      if(gid + 2 < size) {
        sdata[tid] += sdata[tid + 2];
      }
      if(gid + 1 < size) {
        sdata[0] = (sdata[0] + sdata[1]) / size;
      }
    }

    if(threadIdx.x == 0) {
      res[blockDim.x] = sdata[0];
    }
  }

  template<typename T>
  __global__ void normalizeRmse(ftype* val, ftype divisor) {
    const v = val[0];
    if constexpr (std::is_same_v<T, float>) {
      val[0] = __sqrtf(v / divisor);
    }
    else if constexpr (std::is_same_v<T, double>) {
      val[0] = sqrt(v / divisor);
    }
    else {
      static_assert(always_false<T>, "Encountered unexpected ftype");
    }
  }
}

namespace cuda_impl {
  void bceLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / (threadsPerBlock * 2);

    // TODO: res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{loss / nBatches}, y->getDevice(), true);

    if(blocks > 1) {
      // two pass solution
      ftype* tmp; // TODO: Keep this guy in memory for an instance
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      bceLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          tmp, y.getData(), yPred.getData(), y.getDims()[0]);
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }
    else {
      bceLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          res.getData(), y.getData(), yPred.getData(), y.getDims()[0]);
      cudaErrchk(cudaDeviceSynchronize());
    }

    // loss = -loss / nBatches
    divideScalarKernel<<<1, 1>>>(res.getData(), -y.getDims()[0]);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void bceSigmoidLoss(Tensor& res, const Tensor& y, const Tensor& logits) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / (threadsPerBlock * 2);

    if(blocks > 1) {
      // we do two passes at max
      ftype* tmp; 
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      bceSigmoidLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          tmp, y.getData(), logits.getData(), y.getDims()[0]);
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }
    else {
      bceSigmoidLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          res.getData(), y.getData(), logits.getData(), y.getDims()[0]);
      cudaErrchk(cudaDeviceSynchronize());
    }

    // loss = -loss / nBatches
    divideScalarKernel<<<1, 1>>>(res.getData(), -y.getDims()[0]);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropyLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {
    constexpr int maxThreadsPerBlock = 256;
    
    const tensorSize_t stride = y.getDims()[-1];
    const tensorSize_t nSamples = y.getSize() / stride;

    if(y.getSize() * 2 <= maxThreadsPerBlock) {
      int threadsPerBlock = 1;
      while(threadsPerBlock < y.getSize()) threadsPerBlock <<= 1;
      threadsPerBlock = max(1, threadsPerBlock << 1);
      
      const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

      crossEntropyLossKernelOneBlock<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(res.getData(), y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      const int blocks = (y.getSize() + maxThreadsPerBlock - 1) / (maxThreadsPerBlock * 2);

      ftype* tmp;
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      crossEntropyLossKernelOneBlock<<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(ftype)>>>(tmp, y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }

    // loss = -loss / nBatches
    divideScalarKernel<<<1, 1>>>(res.getData(), -nSamples);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropySoftmaxLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // TODO: launch kernel
    cudaErrchk(cudaDeviceSynchronize());
  }

  void rmseLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {
    constexpr int maxThreadsPerBlock = 256;
    
    const auto nSamples = y.getSize();

    if(nSamples * 2 <= maxThreadsPerBlock) {
      int threadsPerBlock = 1;
      while(threadsPerBlock < nSamples) threadsPerBlock <<= 1;
      threadsPerBlock = max(1, threadsPerBlock << 1);
      
      const int blocks = (nSamples + threadsPerBlock - 1) / threadsPerBlock;

      crossEntropyLossKernelOneBlock<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(res.getData(), y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      const int blocks = (nSamples + maxThreadsPerBlock - 1) / (maxThreadsPerBlock * 2);

      ftype* tmp;
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      crossEntropyLossKernelOneBlock<<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(ftype)>>>(tmp, y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }

    normalizeRmse<ftype><<<1, 1>>>(res.getData(), nSamples);
    cudaErrchk(cudaDeviceSynchronize());
  }
}
