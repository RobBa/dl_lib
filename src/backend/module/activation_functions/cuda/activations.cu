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

#include "shared/memory_pool.h"
#include "shared/cuda/common_kernels.cuh"
#include "shared/cuda/common_softmax.cuh"

#include "utility/utils.h"
#include "utility/cuda/cuda_common.cuh"

#include <stdexcept>

using namespace std;

namespace {
  using namespace cuda_impl;

  /**
   * @brief Kernel for forward ReLU function.
   */
  __global__ void reluKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = cudaMax<ftype>(input[gid], 0);
  }

  /**
   * @brief Kernel for forward Leaky-ReLU function.
   */
  __global__ void leakyReluKernel(ftype* const res, const ftype* const input, const ftype eps, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = cudaMax<ftype>(input[gid], eps * input[gid]); // eps < 1
  }

  /**
   * @brief Kernel for forward Sigmoid function.
   */
  __global__ void sigmoidKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = cudaSigmoid<ftype>(input[gid]);
  }

  /**
   * @brief Numerically stable version of softmax kernel. Just as in findMaxKernelOneWarp we assume that stride <= warpsize.
   * Numerical stability comes from computing the maximum values per row, see findMaxKernelOneWarp and argument maxValues.
   */
  template<int maxoffset>
  __global__ void stableSoftmaxKernelOneWarp(ftype* const res, const ftype* const input, const ftype* const maxValues,
                                             const tensorSize_t stride, const tensorSize_t size) {

    const int strideNumber = blockIdx.x * blockDim.y + threadIdx.y; // same as warp number
    const int withinStrideOffset = threadIdx.x; // each warp covers up to 32 elements within a stride
    const int globalIdx = strideNumber * stride + withinStrideOffset;

    const bool isActive = (withinStrideOffset < stride) && (globalIdx < size);

    const auto maxValue = maxValues[strideNumber];
    const ftype expVal = isActive ? stableExp<ftype>(input[globalIdx], maxValue) : 0.0f;

    ftype sum = expVal;
    for(int offset = maxoffset; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, stride); 
    }

    // broadcast thread zero's result
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);

    if(isActive) {
      res[globalIdx] = expVal / sum;
    }
  }

  /**
   * @brief Just like stableSoftmaxKernelOneWarp, but this one works across a whole block, not just a warp.
   * 
   * In this initial version we assume one kernel per stride, to make matters simple to understand.
   */
  __global__ void stableSoftmaxKernelOneBlock(ftype* const res, const ftype* const input, const ftype* const maxValues, const tensorSize_t stride) {
    // Kernel built for one stride per block, blockDim.x is < stride
    assert(blockDim.x < stride); 

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

    for(tensorSize_t offset = blockDim.x; offset > 16; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid < 32) {
      ftype sum = smem[tid];
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if(tid == 0) {
        smem[0] = sum;
      }
    }
    __syncthreads(); // needed because threads > 32 will also use start[0]

    const ftype sum = smem[0];
    res[gid] = expVal / sum;
    if(!doPadding) {
      res[gid + blockDim.x] = expValOffset / sum;
    }
  }

  /**
   * @brief Does the first part of stableSoftmaxKernelOneBlock, namely the sums. Because here again we have a partial sum
   * and assume the stride did not fit into the block entirely, we do a partial sum only. Additionally, write the max-adjusted
   * exp values back to res to prepare for the division.
   */
  template<typename T>
  __global__ void stableSoftmaxLargePass1(ftype* const res, ftype* const partialSums, const ftype* const input, const ftype* const maxValues, 
                                          const tensorSize_t stride, const unsigned int blocksPerStride) {
    const int tid = threadIdx.x;
    const int strideIdx = blockIdx.x / blocksPerStride;
    const unsigned int blockWithinStride = blockIdx.x % blocksPerStride;

    // same logic as in findMaxKernelLargePass1
    const int inputBase = strideIdx * stride + (blockWithinStride << 1) * blockDim.x;
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
    
    // write values to output -> will be nominator in division
    if(localIdx0 < (strideIdx + 1) * stride) {
      res[localIdx0] = expVal0;
    }
    if(localIdx1 < (strideIdx + 1) * stride) {
      res[localIdx1] = expVal1;
    }
    
    // reduce sum
    for(tensorSize_t offset = blockDim.x; offset > 16; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid < 32) {
      ftype sum = smem[tid];
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if(tid == 0) {
        partialSums[blockIdx.x] = sum;
      }
    }
  }

  /**
   * @brief Self explanatory after stableSoftmaxLargePass1. Continues the sum reduce, does not need to write further to res, since 
   * pass 1 already did that for us.
   */
  __global__ void stableSoftmaxLargePass2(ftype* const sums, const ftype* const partialSums, const unsigned int blocksPerStride) {
    // Kernel built for one stride per block, blockDim.x is < stride
    assert(blockDim.x < blocksPerStride); 

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

    for(tensorSize_t offset = blockDim.x >> 1; offset > 16; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid < 32) {
      ftype sum = smem[tid];
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if(tid == 0) { // one block per stride
        sums[blockIdx.x] = sum;
      }
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
    const int blocks = (in.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), in.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void leakyRelu(Tensor& res, const Tensor& in, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), eps, in.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void sigmoid(Tensor& res, const Tensor& in) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    sigmoidKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), in.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  /**
   * @brief Does the softmax computation. Warning: Current implementation can only handle a stride of 
   * at max 512 * 512 = 262144 floating point numbers. If number exceeds this an exception is throws.
   * 
   * For simplicity and for reasons of CUDA efficiency this function is split into 3 segments. 
   * 1. stride <= 32 -> we use warp level.
   * 2. stride > 32 && stride < 512 -> we can fit one stride into one block.
   * 3. stride > 512 -> we have to use two-stage kernel cascading for parallel reduction.
   */
  void softmax(Tensor& res, const Tensor& in) {
    const tensorSize_t stride = static_cast<tensorSize_t>(in.getDims().get(-1));
    const int nStrides = in.getSize() / stride;

    ftype* const maxValues = mempool::tensorPool.request(Device::CUDA, nStrides);

    constexpr int warpSize = 32;
    if(stride <= warpSize) {
      assert(utility::DeviceProperties::getWarpSize() == 32);

      // each warp does one stride
      constexpr int threadsPerBlock = 256;
      constexpr int warpsPerBlock = threadsPerBlock / 32;
      constexpr dim3 blockDims(32, warpsPerBlock);
      const int blocks = (nStrides + warpsPerBlock - 1) / warpsPerBlock;

      if(stride <= 2) {
        findMaxKernelOneWarp<1><<<blocks, blockDims>>>(maxValues, in.getData(), stride, nStrides);
      }
      else if(stride <= 4) {
        findMaxKernelOneWarp<2><<<blocks, blockDims>>>(maxValues, in.getData(), stride, nStrides);
      }
      else if(stride <= 8) {
        findMaxKernelOneWarp<4><<<blocks, blockDims>>>(maxValues, in.getData(), stride, nStrides);
      }
      else if(stride <= 16) {
        findMaxKernelOneWarp<8><<<blocks, blockDims>>>(maxValues, in.getData(), stride, nStrides);
      }
      else if(stride <= 32) {
        findMaxKernelOneWarp<16><<<blocks, blockDims>>>(maxValues, in.getData(), stride, nStrides);
      }
      
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      if(stride <= 2) {
        stableSoftmaxKernelOneWarp<1><<<blocks, blockDims>>>(res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 4) {
        stableSoftmaxKernelOneWarp<2><<<blocks, blockDims>>>(res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 8) {
        stableSoftmaxKernelOneWarp<4><<<blocks, blockDims>>>(res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 16) {
        stableSoftmaxKernelOneWarp<8><<<blocks, blockDims>>>(res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 32) {
        stableSoftmaxKernelOneWarp<16><<<blocks, blockDims>>>(res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif
    }
    else if (stride <= 512) {
      constexpr int stridesPerBlock = 1; // if multiple strides per block allowed (adapt kernels!): max(maxThreadsPerBlock / stride, 1);

      // threads per block needs to be power of 2 for reduction to resolve cleanly
      int threadsPerBlock = 1;
      while(threadsPerBlock < stride) threadsPerBlock <<= 1;
      threadsPerBlock >>= 1;

      const int blocks = (nStrides + stridesPerBlock - 1) / stridesPerBlock; // gerneralized version iff multiple strides per block allowed

      findMaxKernelOneBlock<<<blocks, threadsPerBlock, (threadsPerBlock << 1) * sizeof(ftype)>>>(maxValues, in.getData(), stride);
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      stableSoftmaxKernelOneBlock<<<blocks, threadsPerBlock, (threadsPerBlock << 1) * sizeof(ftype)>>>(res.getData(), in.getData(), maxValues, 
                                                                                                           stride);
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif
    }
    else {
      // stride does not fit into one block. We employ a 2 pass system, where pass one does a partial 
      // reduction, and pass two does a reduction over the partial reductions.
    
      // each block handles up to 512 elements (2 * 256 threads)
      constexpr int maxThreadsPerBlock = 256;
      constexpr int elemsPerBlock = 2 * maxThreadsPerBlock; // constant folding
      const unsigned int blocksPerStride = (stride + elemsPerBlock - 1) / elemsPerBlock;
      assert_debug(blocksPerStride <= 512, "Stride too large for two-pass reduction");
      
      const int totalBlocks = nStrides * blocksPerStride;

      // intermediate max values: one per block per stride
      const tensorSize_t nPartialMax = totalBlocks;
      ftype* partialMaxValues = mempool::tensorPool.request(Device::CUDA, nPartialMax);

      // pass 1: reduce each chunk of 512 elements to one partial max
      // launch blocksPerStride blocks per stride
      findMaxKernelLargePass1<<<totalBlocks, maxThreadsPerBlock, (maxThreadsPerBlock << 1) * sizeof(ftype)>>>(
                                partialMaxValues, in.getData(), stride, blocksPerStride);
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      // pass 2: reduce partial maxes to one max per stride
      int threadsPass2 = 1;
      while(threadsPass2 < blocksPerStride) threadsPass2 <<= 1; // threadsPass2 needs to be power of 2
      threadsPass2 = max(1, threadsPass2 >> 1);
      
      findMaxKernelLargePass2<<<nStrides, threadsPass2, threadsPass2 * sizeof(ftype)>>>(maxValues, partialMaxValues, blocksPerStride);
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      // same two-pass structure for the sum
      ftype* partialSums = mempool::tensorPool.request(Device::CUDA, nPartialMax);

      stableSoftmaxLargePass1<ftype><<<totalBlocks, maxThreadsPerBlock, (maxThreadsPerBlock << 1) * sizeof(ftype)>>>(
                        res.getData(), partialSums, in.getData(), maxValues, stride, blocksPerStride);
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      // pass 2: reduce partial sums
      stableSoftmaxLargePass2<<<nStrides, threadsPass2, threadsPass2 * sizeof(ftype)>>>(partialSums, partialSums, blocksPerStride);
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      // final division pass: divide each exp value by its stride's sum
      const int nBlocksDivision = (in.getSize() + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
      divideKernel<<<nBlocksDivision, maxThreadsPerBlock>>>(res.getData(), partialSums, stride, in.getSize());
      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif

      mempool::tensorPool.giveback(partialMaxValues, Device::CUDA, nPartialMax);
      mempool::tensorPool.giveback(partialSums, Device::CUDA, nPartialMax);
    }

    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
    
    mempool::tensorPool.giveback(maxValues, Device::CUDA, nStrides);
  }
}