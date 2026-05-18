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
#include "utility/macros.h"

#include <stdexcept>

using namespace std;

namespace {
  /**
   * @brief Kernel for forward ReLU function.
   */
  __global__ void reluKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    constexpr ftype zero = 0;
    res[gid] = fmaxf(input[gid], zero);
  }

  /**
   * @brief Kernel for forward Leaky-ReLU function.
   */
  __global__ void leakyReluKernel(ftype* const res, const ftype* const input, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = fmaxf(input[gid], eps * input[gid]); // eps < 1
  }

  /**
   * @brief Single sigmoid computation.
   */
  __device__ __forceinline__ ftype sigmoid(ftype x) {
      ftype z = expf(-fabsf(x));
      ftype s = 1.0f / (1.0f + z);
      return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
  }

  /**
   * @brief Kernel for forward Sigmoid function.
   */
  __global__ void sigmoidKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = sigmoid(input[gid]);
  }

  /**
   * @brief Reduction kernel that computes the maximum within the size of 2 * warpsize at maximum.
   */
  template<int maxoffset>
  __forceinline__ __device__ void warpMaxReduce(volatile ftype* const input, const tensorSize_t stride, const int offset) {
    // TODO: warp shuffle for newer architectures
    if(maxoffset == 32) {
      if(offset + 32 < stride) input[offset] = max(input[offset], input[offset + 32]);
    }
    if(maxoffset >= 16) {
      if(offset + 16 < stride) input[offset] = max(input[offset], input[offset + 16]);
    }
    if(maxoffset >= 8) {
      if(offset + 8 < stride) input[offset] = max(input[offset], input[offset + 8]);
    }
    if(maxoffset >= 4) {
      if(offset + 4 < stride) input[offset] = max(input[offset], input[offset + 4]);
    }
    if(maxoffset >= 2) {
      if(offset + 2 < stride) input[offset] = max(input[offset], input[offset + 2]);
    }
    if(maxoffset >= 1) {
      if(offset + 1 < stride) input[offset] = max(input[offset], input[offset + 1]);
    }
  }

  /**
   * @brief Reduction kernel that computes the sum over an array within the size of 2 * warpsize at maximum.
   */
  template<int maxoffset>
  __forceinline__ __device__ void warpSumReduce(volatile ftype* const input, const tensorSize_t stride, const int offset) {
    // TODO: warp shuffle for newer architectures
    if(maxoffset == 32) {
      if(offset + 32 < stride) input[offset] += input[offset + 32];
    }
    if(maxoffset >= 16) {
      if(offset + 16 < stride) input[offset] += input[offset + 16];
    }
    if(maxoffset >= 8) {
      if(offset + 8 < stride) input[offset] += input[offset + 8];
    }
    if(maxoffset >= 4) {
      if(offset + 4 < stride) input[offset] += input[offset + 4];
    }
    if(maxoffset >= 2) {
      if(offset + 2 < stride) input[offset] += input[offset + 2];
    }
    if(maxoffset >= 1) {
      if(offset + 1 < stride) input[offset] += input[offset + 1];
    }
  }

  /**
   * @brief For the softmax implementations.
   */
  template<typename T>
  __forceinline__ __device__ ftype stableExp(const ftype x, const ftype maxValue) {
    if constexpr (std::is_same_v<T, float>) {
      return expf(x - maxValue);
    }
    else if constexpr (std::is_same_v<T, double>) {
      return exp(x - maxValue);
    }
    else {
      static_assert(always_false<T>, "ftype encountered unexpected type");
    }
  } 
  
  /**
   * @brief Here we find the maximum within 'stride'. Assumption: One warp does exactly one element of stride!
   * Reduction via warp reduce. res has the maximum values stored.
   */
  template<int maxoffset>
  __global__ void findMaxKernelOneWarp(ftype* const res, const ftype* const input, const tensorSize_t stride, const tensorSize_t size) {
    assert(blockDim.x % 32 == 0);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    int tid = threadIdx.x;
    extern __shared__ ftype smem[];
    smem[tid] = input[gid];
    __syncthreads();

    volatile ftype* const start = smem + (tid / stride) * stride;
    const int offset = gid % stride;
    warpMaxReduce<maxoffset>(start, stride, offset);

    // one warp reduces one 'stride'
    if(offset == 0) {
      res[tid / 32] = smem[tid];
    }
  }

  /**
   * @brief Numerically stable version of softmax kernel. Just as in findMaxKernelOneWarp we assume that stride <= 2 * warpsize.
   * Numerical stability comes from computing the maximum values per row, see findMaxKernelOneWarp and argument maxValues.
   */
  template<typename T, int maxoffset>
  __global__ void stableSoftmaxKernelOneWarp(ftype* const res, const ftype* const input, const ftype* const maxValues,
                                             const tensorSize_t stride, const tensorSize_t size) {
    assert(blockDim.x % 32 == 0);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    const auto maxValue = maxValues[strideOffset];
    ftype expVal = stableExp<ftype>(input[gid], maxValue);

    int tid = threadIdx.x;
    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>
    smem[tid] = expVal;
    __syncthreads();

    volatile ftype* const start = smem + (tid / stride) * stride;
    const int offset = gid % stride;
    warpSumReduce<maxoffset>(start, stride, offset);

    res[gid] = expVal / start[0];
  }

  /**
   * @brief Like findMaxKernelOneWarp. The difference now is that the input size can be much larger. stride is 
   * 64 < stride <= threadsPerBlock. res has the maximum values stored.
   * 
   * In this initial version we assume one kernel per stride, to make matters simple to understand.
   */
  __global__ void findMaxKernelOneBlock(ftype* const res, const ftype* const input, const tensorSize_t stride) {
    assert_debug(blockDim.x / stride == 0, "Kernel built for one stride per block, blockDim.x is < stride"); 

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * stride + tid;

    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>

    const tensorSize_t maxIdx = tid + blockDim.x;
    const bool doPadding = maxIdx >= stride;
    if(doPadding) {
      smem[tid] = input[gid];
    }
    else {
      smem[tid] = max(input[gid], input[gid + blockDim.x]);
    }
    __syncthreads();

    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] = max(smem[tid], smem[tid + offset]);
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) {
      start[tid] = max(start[tid], start[tid + 32]);
      start[tid] = max(start[tid], start[tid + 16]);
      start[tid] = max(start[tid], start[tid + 8]);
      start[tid] = max(start[tid], start[tid + 4]);
      start[tid] = max(start[tid], start[tid + 2]);
      start[tid] = max(start[tid], start[tid + 1]);
    }

    if(tid == 0) { // one block per stride
      res[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Just like stableSoftmaxKernelOneWarp, but this one works across a whole block, not just a warp.
   * 
   * In this initial version we assume one kernel per stride, to make matters simple to understand.
   */
  template<typename T>
  __global__ void stableSoftmaxKernelOneBlock(ftype* const res, const ftype* const input, const ftype* const maxValues, const tensorSize_t stride) {
    assert_debug(blockDim.x / stride == 0, "Kernel built for one stride per block, blockDim.x is < stride"); 

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * stride + tid;

    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>
    const auto maxValue = maxValues[blockIdx.x];
    ftype expVal = stableExp<ftype>(input[gid], maxValue);
    smem[tid] = expVal;

    const tensorSize_t maxIdx = tid + blockDim.x;
    const bool doPadding = maxIdx >= stride;

    ftype expValOffset = 0;
    if(!doPadding) { // some threads will be idle here
      expValOffset = stableExp<ftype>(input[gid + blockDim.x], maxValue);
    }

    smem[maxIdx] = expValOffset;
    __syncthreads();

    for(tensorSize_t offset = blockDim.x; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) {
      start[tid] += start[tid + 32];
      start[tid] += start[tid + 16];
      start[tid] += start[tid + 8];
      start[tid] += start[tid + 4];
      start[tid] += start[tid + 2];
      start[tid] += start[tid + 1];
    }
    __syncthreads();

    res[gid] = expVal / start[0];
    if(!doPadding) {
      res[gid + blockDim.x] = expValOffset / start[0];
    }
  }

  /**
   * @brief Like findMaxKernelOneBlock, but finding partial maximum. This is the case when the stride is too
   * large to fit in one block.
   */
  __global__ void findMaxKernelLargePass1(ftype* const partialMaxValues, const ftype* const input, 
                                          const tensorSize_t stride, const int blocksPerStride) {
    const int tid = threadIdx.x;
    const int strideIdx = blockIdx.x / blocksPerStride;
    const int blockWithinStride = blockIdx.x % blocksPerStride;

    // block 0 within stride handles elements [0, 2*blockDim.x), block 1 within stride handles elements [2*blockDim.x, 4*blockDim.x), ...
    const int inputBase = strideIdx * stride + blockWithinStride * 2 * blockDim.x; 
    
    extern __shared__ ftype smem[];
    const tensorSize_t localIdx0 = inputBase + tid;
    const tensorSize_t localIdx1 = inputBase + tid + blockDim.x;
    
    // localIdx0 < (strideIdx + 1) * stride <- checks whether thread idx exceeds bounds of this stride; one stride per block at max
    smem[tid] = (localIdx0 < (strideIdx + 1) * stride) ? input[localIdx0] : -INFINITY;
    smem[tid + blockDim.x] = (localIdx1 < (strideIdx + 1) * stride) ? input[localIdx1] : -INFINITY;
    __syncthreads();
    
    // same reduction as findMaxKernelOneBlock from here
    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset){
        smem[tid] = max(smem[tid], smem[tid + offset]);
      } 
      __syncthreads();
    }

    volatile ftype* start = smem;
    if(tid < 32) {
      start[tid] = max(start[tid], start[tid + 32]);
      start[tid] = max(start[tid], start[tid + 16]);
      start[tid] = max(start[tid], start[tid + 8]);
      start[tid] = max(start[tid], start[tid + 4]);
      start[tid] = max(start[tid], start[tid + 2]);
      start[tid] = max(start[tid], start[tid + 1]);
    }

    if(tid == 0) {
      partialMaxValues[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Self explanatory following findMaxKernelLargePass1. Assumption: All remaining max values do fit into 
   * one single block now -> we launch one block per stride this time.
   */
  __global__ void findMaxKernelLargePass2(ftype* const maxValues, const ftype* const partialMaxValues, const tensorSize_t blocksPerStride) {
    assert_debug(blockDim.x / blocksPerStride == 0, "Kernel built for one stride per block, blockDim.x is < stride"); 

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blocksPerStride + tid;

    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>

    const tensorSize_t maxIdx = tid + blockDim.x;
    const bool doPadding = maxIdx >= blocksPerStride;
    if(doPadding) {
      smem[tid] = partialMaxValues[gid];
    }
    else {
      smem[tid] = max(partialMaxValues[gid], partialMaxValues[gid + blockDim.x]);
    }
    __syncthreads();

    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] = max(smem[tid], smem[tid + offset]);
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) {
      if(32 < blockDim.x * 2) start[tid] = max(start[tid], start[tid + 32]);
      if(16 < blockDim.x * 2) start[tid] = max(start[tid], start[tid + 16]);
      if(8  < blockDim.x * 2) start[tid] = max(start[tid], start[tid + 8]);
      if(4  < blockDim.x * 2) start[tid] = max(start[tid], start[tid + 4]);
      if(2  < blockDim.x * 2) start[tid] = max(start[tid], start[tid + 2]);
      if(1  < blockDim.x * 2) start[tid] = max(start[tid], start[tid + 1]);
    }

    if(tid == 0) { // one block per stride
      maxValues[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Does the first part of stableSoftmaxKernelOneBlock, namely the sums. Because here again we have a partial sum
   * and assume the stride did not fit into the block entirely, we do a partial sum only. Additionally, write the max-adjusted
   * exp values back to res to prepare for the division.
   */
  template<typename T>
  __global__ void stableSoftmaxLargePass1(ftype* const res, ftype* const partialSums, const ftype* const input, const ftype* const maxValues, 
                                          const tensorSize_t stride, const int blocksPerStride) {
    const int tid = threadIdx.x;
    const int strideIdx = blockIdx.x / blocksPerStride;
    const int blockWithinStride = blockIdx.x % blocksPerStride;

    // same logic as in findMaxKernelLargePass1
    const int inputBase = strideIdx * stride + blockWithinStride * 2 * blockDim.x;
    const ftype maxValue = maxValues[strideIdx];
    
    extern __shared__ ftype smem[];
    const tensorSize_t localIdx0 = inputBase + tid;
    const tensorSize_t localIdx1 = inputBase + tid + blockDim.x;
    
    // same logic as in findMaxKernelLargePass1
    ftype expVal0 = (localIdx0 < (strideIdx + 1) * stride) ? stableExp<T>(input[localIdx0], maxValue) : 0.0f;
    ftype expVal1 = (localIdx1 < (strideIdx + 1) * stride) ? stableExp<T>(input[localIdx1], maxValue) : 0.0f;
    
    smem[tid] = expVal0;
    smem[tid + blockDim.x] = expVal1;
    __syncthreads();
    
    // write exp values to output
    if(localIdx0 < (strideIdx + 1) * stride) {
      res[localIdx0] = expVal0;
    }
    if(localIdx1 < (strideIdx + 1) * stride) {
      res[localIdx1] = expVal1;
    }
    
    // reduce sum
    for(tensorSize_t offset = blockDim.x; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    volatile ftype* start = smem;
    if(tid < 32) {
      start[tid] += start[tid + 32];
      start[tid] += start[tid + 16];
      start[tid] += start[tid + 8];
      start[tid] += start[tid + 4];
      start[tid] += start[tid + 2];
      start[tid] += start[tid + 1];
    }
    //__syncthreads();

    if(tid == 0) {
      partialSums[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Self explanatory after stableSoftmaxLargePass1. Continues the sum reduce, does not need to write further to res, since 
   * pass 1 already did that for us.
   */
  __global__ void stableSoftmaxLargePass2(ftype* const sums, const ftype* const partialSums, const tensorSize_t blocksPerStride) {
    assert_debug(blockDim.x / blocksPerStride == 0, "Kernel built for one stride per block, blockDim.x is < stride"); 

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blocksPerStride + tid;

    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>

    const tensorSize_t maxIdx = tid + blockDim.x;
    const bool doPadding = maxIdx >= blocksPerStride;
    if(doPadding) {
      smem[tid] = partialSums[gid];
    }
    else {
      smem[tid] = partialSums[gid] + partialSums[gid + blockDim.x];
    }
    __syncthreads();

    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) {
      if(32 < blockDim.x * 2) start[tid] += start[tid + 32];
      if(16 < blockDim.x * 2) start[tid] += start[tid + 16];
      if(8  < blockDim.x * 2) start[tid] += start[tid + 8];
      if(4  < blockDim.x * 2) start[tid] += start[tid + 4];
      if(2  < blockDim.x * 2) start[tid] += start[tid + 2];
      if(1  < blockDim.x * 2) start[tid] += start[tid + 1];
    }
    //__syncthreads();

    if(tid == 0) { // one block per stride
      sums[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Simple kernel doing the division of softmax in the case of a large stride.
   */
  __global__ void divideKernel(ftype* const res, const ftype* const sums, const tensorSize_t stride, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    res[gid] /= sums[gid / stride];
  }
}

namespace cuda_impl {
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

  /**
   * @brief Does the softmax computation. Warning: Current implementation can only handle a stride of 
   * at max 512 * 512 = 262144 floating point numbers. If number exceeds this an exception is throws.
   * 
   * For simplicity and for reasons of CUDA efficiency this function is split into 3 segments. 
   * 1. stride <= 64 -> we use warp level.
   * 2. stride > 64 && stride < 512 -> we can fit one stride into one block.
   * 3. stride > 512 -> we have to use two-stage kernel cascading for parallel reduction.
   */
  void softmax(Tensor& res, const Tensor& in) {
    const tensorSize_t stride = static_cast<tensorSize_t>(in.getDims().get(-1));
    if(stride == 1) {
      __throw_invalid_argument("Attemped softmax of one element");
    }

    const int nStrides = in.getSize() / stride;

    // TODO: use some static struct here to prevent this guy from keeping on re-allocating memory
    ftype* maxValues;
    const tensorSize_t nMaxValues = nStrides;
    cudaErrchk(cudaMalloc(&maxValues, nMaxValues * sizeof(ftype)));

    static const auto warpSizeT2 = 2 * DeviceProperties::getWarpSize(); // TODO: can this be a problem in a multi-GPU setting?
    if(stride <= warpSizeT2) {
      assert(DeviceProperties::getWarpSize() == 32);

      const int threadsPerBlock = 256;
      const int blocks = (in.getSize() + threadsPerBlock - 1) / threadsPerBlock;

      if(stride == 2) {
        findMaxKernelOneWarp<1> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 4) {
        findMaxKernelOneWarp<2> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 8) {
        findMaxKernelOneWarp<4> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 16) {
        findMaxKernelOneWarp<8> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 32) {
        findMaxKernelOneWarp<16> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 64) {
        findMaxKernelOneWarp<32> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      cudaErrchk(cudaDeviceSynchronize());

      if(stride == 2) {
        stableSoftmaxKernelOneWarp<ftype, 1> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 4) {
        stableSoftmaxKernelOneWarp<ftype, 2> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 8) {
        stableSoftmaxKernelOneWarp<ftype, 4> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 16) {
        stableSoftmaxKernelOneWarp<ftype, 8> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 32) {
        stableSoftmaxKernelOneWarp<ftype, 16> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 64) {
        stableSoftmaxKernelOneWarp<ftype, 32> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      cudaErrchk(cudaDeviceSynchronize());
    }
    else if (stride <= 512) {
      constexpr int maxThreadsPerBlock = 256;
      assert_debug(stride <= 2 * maxThreadsPerBlock, "If you adapt maxThreadsPerBlock you also have to adapt the limit in the if-clause above");

      constexpr int stridesPerBlock = 1; // if multiple strides per block allowed (adapt kernels!): max(maxThreadsPerBlock / stride, 1);

      // threads per block needs to be power of 2 for reduction to resolve cleanly
      int paddedStride = 1;
      while(paddedStride < stride) paddedStride <<= 1;
      int threadsPerBlock = paddedStride / 2;

      const int blocks = (nStrides + stridesPerBlock - 1) / stridesPerBlock; // gerneralized version iff multiple strides per block allowed

      findMaxKernelOneBlock<<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride);
      cudaErrchk(cudaDeviceSynchronize());

      stableSoftmaxKernelOneBlock<ftype><<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(res.getData(), in.getData(), maxValues, 
                                                                                                           stride);
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      // stride does not fit into one block. We employ a 2 pass system, where pass one does a partial 
      // reduction, and pass two does a reduction over the partial reductions.
    
      // each block handles up to 512 elements (2 * 256 threads)
      constexpr int maxThreadsPerBlock = 256;
      constexpr int elemsPerBlock = 2 * maxThreadsPerBlock; // constant folding
      const int blocksPerStride = (stride + elemsPerBlock - 1) / elemsPerBlock;
      assert_debug(blocksPerStride <= 512, "Stride too large for two-pass reduction");
      
      const int totalBlocks = nStrides * blocksPerStride;

      // intermediate max values: one per block per stride
      ftype* partialMaxValues;
      const tensorSize_t nPartialMax = totalBlocks;
      cudaErrchk(cudaMalloc(&maxValues, nStrides * sizeof(ftype)));
      cudaErrchk(cudaMalloc(&partialMaxValues, nPartialMax * sizeof(ftype)));

      // pass 1: reduce each chunk of 512 elements to one partial max
      // launch blocksPerStride blocks per stride
      findMaxKernelLargePass1<<<totalBlocks, maxThreadsPerBlock, 2 * maxThreadsPerBlock * sizeof(ftype)>>>(
                                partialMaxValues, in.getData(), stride, blocksPerStride);
      cudaErrchk(cudaDeviceSynchronize());

      // pass 2: reduce partial maxes to one max per stride
      int threadsPass2 = 1;
      while(threadsPass2 < blocksPerStride) threadsPass2 <<= 1; // threadsPass2 needs to be power of 2
      threadsPass2 /= 2;
      
      findMaxKernelLargePass2<<<nStrides, threadsPass2, 2 * threadsPass2 * sizeof(ftype)>>>(maxValues, partialMaxValues, blocksPerStride);
      cudaErrchk(cudaDeviceSynchronize());

      // softmax: same two-pass structure for the sum
      ftype* partialSums;
      cudaErrchk(cudaMalloc(&partialSums, nPartialMax * sizeof(ftype)));

      stableSoftmaxLargePass1<ftype><<<totalBlocks, maxThreadsPerBlock, 2 * maxThreadsPerBlock * sizeof(ftype)>>>(
                        res.getData(), partialSums, in.getData(), maxValues, stride, blocksPerStride);
      cudaErrchk(cudaDeviceSynchronize());

      // pass 2: reduce partial sums
      stableSoftmaxLargePass2<<<nStrides, threadsPass2, 2 * threadsPass2 * sizeof(ftype)>>>(partialSums, partialSums, blocksPerStride);
      cudaErrchk(cudaDeviceSynchronize());

      // final division pass: divide each exp value by its stride's sum
      const int nBlocksDivision = (in.getSize() + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
      divideKernel<<<nBlocksDivision, maxThreadsPerBlock>>>(res.getData(), partialSums, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      cudaErrchk(cudaFree(partialMaxValues));
      cudaErrchk(cudaFree(partialSums));
    }
    cudaErrchk(cudaFree(maxValues));
  }
}