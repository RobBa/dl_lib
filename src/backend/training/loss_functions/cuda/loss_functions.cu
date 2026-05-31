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

#include "utility/utils.h"
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
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ ftype smem[];
    const ftype tmp = gid < size ? bce<ftype>(y[gid], ypred[gid]) : 0;
    smem[tid] = tmp;
    __syncthreads();

    for(tensorSize_t offset = blockDim.x / 2; offset > 32; offset >>= 1){
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
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
        sdata[tid] += sdata[tid + 1];
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
      return cudaMax<T>(logit, zero) - logit * y + __logf(1 + __expf(-abs(logit)));
    }
    else if constexpr (std::is_same_v<T, double>) {
      return cudaMax<T>(logit, zero) - logit * y + log(1 + exp(-abs(logit)));
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
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;    
    const int tid = threadIdx.x;

    extern __shared__ ftype smem[];

    // pre-load first round
    const ftype tmp = gid < size ? bceSimplified<ftype>(y[gid], logits[gid]) : 0;
    smem[tid] = tmp;
    __syncthreads();

    for(tensorSize_t offset = blockDim.x / 2; offset > 32; offset >>= 1){
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
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
        sdata[tid] += sdata[tid + 1];
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
   * @brief Cross-entropy reduction over a full block. Covers two times blockDim.x.
   */
  __global__ void crossEntropyLossKernelOneBlock(ftype* const res, const ftype* const y, const ftype* const yPred, const tensorSize_t size) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ ftype smem[];

    const ftype tmp = gid < size ? crossEntropy<ftype>(y[gid], yPred[gid]) : 0;
    smem[tid] = tmp;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
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
        sdata[tid] += sdata[tid + 1];
      }
    }

    if(threadIdx.x == 0) {
      res[blockIdx.x] = sdata[0];
    }
  }

  /**
   * @brief Softmax cross-entropy loss kernel. One block per sample.
   */

  template<typename T>
  __global__ void crossEntropySoftmaxLossKernel(ftype* const perSampleLoss, const ftype* const y, const ftype* const logits,
                                                const ftype* const maxValues, const tensorSize_t stride) {
    const int tid  = threadIdx.x;
    const int tid2 = tid + blockDim.x;
    const int sampleBase = blockIdx.x * stride;
    const ftype maxV = maxValues[blockIdx.x];

    extern __shared__ ftype smem[];

    const bool inBounds0 = (tid < stride);
    const bool inBounds1 = (tid2 < stride);

    constexpr ftype zero = 0;
    ftype e0  = inBounds0 ? stableExp<T>(logits[sampleBase + tid],  maxV) : zero;
    ftype yz0 = inBounds0 ? y[sampleBase + tid]  * logits[sampleBase + tid]  : zero;
    ftype e1  = inBounds1 ? stableExp<T>(logits[sampleBase + tid2], maxV) : zero;
    ftype yz1 = inBounds1 ? y[sampleBase + tid2] * logits[sampleBase + tid2] : zero;

    // Phase 1: reduce exp sum across stride (smem has 2 * blockDim.x slots)
    smem[tid]  = e0;
    smem[tid2] = e1;
    __syncthreads();

    for(int offset = blockDim.x; offset >= 1; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    const ftype expSum = smem[0];

    // reduce y*z dot product (y is one-hot, so at most one non-zero per sample)
    smem[tid] = yz0 + yz1;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset >= 1; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid == 0) {
      ftype logExpSum;
      if constexpr (std::is_same_v<T, float>) {
        logExpSum = __logf(expSum);
      }
      else if constexpr (std::is_same_v<T, double>) {
        logExpSum = log(expSum);
      }
      else {
        static_assert(always_false<T>, "Unexpected value for ftype encountered");
      }
      perSampleLoss[blockIdx.x] = maxV + logExpSum - smem[0];
    }
  }

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
    const ftype tmp = gid < size ? diffPow(y[gid], yPred[gid]) : 0;
    smem[tid] = tmp;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
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
        sdata[tid] += sdata[tid + 1];
      }
    }

    if(threadIdx.x == 0) {
      res[blockIdx.x] = sdata[0];
    }
  }

  template<typename T>
  __global__ void normalizeRmse(ftype* const val, ftype divisor) {
    const ftype v = val[0];
    if constexpr (std::is_same_v<T, float>) {
      val[0] = sqrtf(v / divisor);
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
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    if(blocks > 1) {
      // two pass solution
      ftype* tmp; // TODO: Keep this guy in memory for an instance
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      bceLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          tmp, y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }
    else {
      bceLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          res.getData(), y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }

    // loss = -loss / nBatches
    divideScalarKernel<<<1, 1>>>(res.getData(), -1 * static_cast<ftype>(y.getDims()[0]));
    cudaErrchk(cudaDeviceSynchronize());
  }

  void bceSigmoidLoss(Tensor& res, const Tensor& y, const Tensor& logits) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    if(blocks > 1) {
      // we do two passes at max
      ftype* tmp; 
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      bceSigmoidLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          tmp, y.getData(), logits.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }
    else {
      bceSigmoidLossKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(
          res.getData(), y.getData(), logits.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }

    // loss = -loss / nBatches
    divideScalarKernel<<<1, 1>>>(res.getData(), -1 * static_cast<ftype>(y.getDims()[0]));
    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropyLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {    
    if(y.getSize() <= 256) {
      int threadsPerBlock = 1;
      while(threadsPerBlock < y.getSize()) threadsPerBlock <<= 1; // < 512 threads
      
      crossEntropyLossKernelOneBlock<<<1, threadsPerBlock, y.getSize() * sizeof(ftype)>>>(res.getData(), y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      constexpr int threadsPerBlock = 256;
      const int blocks = (y.getSize() + threadsPerBlock - 1) / threadsPerBlock;

      ftype* tmp;
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      crossEntropyLossKernelOneBlock<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(tmp, y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }

    // loss = -loss / nBatches
    const tensorSize_t stride = y.getDims()[-1];
    const tensorSize_t nSamples = y.getSize() / stride;
    divideScalarKernel<<<1, 1>>>(res.getData(), -1 * static_cast<ftype>(nSamples));
    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropySoftmaxLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {
    const tensorSize_t stride   = static_cast<tensorSize_t>(yPred.getDims().get(-1));
    const tensorSize_t nSamples = yPred.getSize() / stride;

    ftype* maxValues;
    cudaErrchk(cudaMalloc(&maxValues, nSamples * sizeof(ftype)));

    // Find per-sample max values for numerical stability (mirrors softmax dispatch)
    static const auto warpSizeT2 = 2 * DeviceProperties::getWarpSize();
    if(stride <= warpSizeT2) {
      constexpr int threadsPerBlock = 256;
      const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

      if(stride == 2)
        findMaxKernelOneWarp<1><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride, yPred.getSize());
      else if(stride <= 4)
        findMaxKernelOneWarp<2><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride, yPred.getSize());
      else if(stride <= 8)
        findMaxKernelOneWarp<4><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride, yPred.getSize());
      else if(stride <= 16)
        findMaxKernelOneWarp<8><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride, yPred.getSize());
      else if(stride <= 32)
        findMaxKernelOneWarp<16><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride, yPred.getSize());
      else
        findMaxKernelOneWarp<32><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride, yPred.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else if(stride <= 512) {
      int threadsPerBlock = 1;
      while(threadsPerBlock < stride) threadsPerBlock <<= 1;
      threadsPerBlock /= 2;

      findMaxKernelOneBlock<<<nSamples, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(maxValues, yPred.getData(), stride);
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      cudaErrchk(cudaFree(maxValues));
      __throw_invalid_argument("crossEntropySoftmaxLoss: stride > 512 not yet supported on CUDA");
    }

    // one block per sample, each thread covers up to 2 elements
    int threadsPerBlock = 1;
    while(threadsPerBlock * 2 < stride) threadsPerBlock <<= 1;
    threadsPerBlock = max(1, threadsPerBlock);

    ftype* perSampleLoss;
    cudaErrchk(cudaMalloc(&perSampleLoss, nSamples * sizeof(ftype)));

    crossEntropySoftmaxLossKernel<ftype><<<nSamples, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(
      perSampleLoss, y.getData(), yPred.getData(), maxValues, stride);
    cudaErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<ftype> lossPtr(perSampleLoss);
    thrust::device_ptr<ftype> resPtr(res.getData());
    resPtr[0] = thrust::reduce(lossPtr, lossPtr + nSamples, static_cast<ftype>(0), thrust::plus<ftype>())
                / static_cast<ftype>(nSamples);

    cudaErrchk(cudaFree(perSampleLoss));
    cudaErrchk(cudaFree(maxValues));
  }

  void rmseLoss(Tensor& res, const Tensor& y, const Tensor& yPred) {    
    const auto nSamples = y.getSize();

    if(nSamples <= 256) {
      int threadsPerBlock = 1;
      while(threadsPerBlock < nSamples) threadsPerBlock <<= 1; // < 512 threads
      
      rmseKernelOneBlock<<<1, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(res.getData(), y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      constexpr int threadsPerBlock = 256;
      const int blocks = (nSamples + threadsPerBlock - 1) / threadsPerBlock;

      ftype* tmp;
      cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

      rmseKernelOneBlock<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(tmp, y.getData(), yPred.getData(), y.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      // do a sum over the residual array
      thrust::device_ptr<ftype> tmpPtr(tmp);
      thrust::device_ptr<ftype> resPtr(res.getData());
      resPtr[0] = thrust::reduce(tmpPtr, tmpPtr + blocks, static_cast<ftype>(0.0f), thrust::plus<ftype>());

      cudaErrchk(cudaFree(tmp));
    }

    normalizeRmse<ftype><<<1, 1>>>(res.getData(), static_cast<ftype>(nSamples));
    cudaErrchk(cudaDeviceSynchronize());
  }
}
