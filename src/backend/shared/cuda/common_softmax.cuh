/**
 * @file common_softmax.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Common kernels for softmax.
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "common_kernels.cuh"

#include "shared/global_params.h"
#include "utility/utils.h"

namespace cuda_impl {

  /**
   * @brief Basic helper.
   */
  template<typename T>
  __forceinline__ __device__ ftype stableExp(const ftype x, const ftype maxVal) {
    if constexpr (std::is_same_v<T, float>) {
      return __expf(x - maxVal);
    }
    else if constexpr (std::is_same_v<T, double>) {
      return exp(x - maxVal);
    }
    else {
      static_assert(always_false<T>, "ftype encountered unexpected type");
    }
  }

  /**
   * @brief Reduction kernel that computes the maximum within the size of 2 * warpsize at maximum.
   */
  template<int maxoffset>
  __forceinline__ __device__ void warpMaxReduce(volatile ftype* const input, const tensorSize_t stride, const int offset) {
    static_assert(maxoffset > 0 && maxoffset <= 32, "Invalid value for template");
    // TODO: warp shuffle for newer architectures
    if(maxoffset == 32) {
      if(offset + 32 < stride) input[offset] = cudaMax<ftype>(input[offset], input[offset + 32]);
    }
    if(maxoffset >= 16) {
      if(offset + 16 < stride) input[offset] = cudaMax<ftype>(input[offset], input[offset + 16]);
    }
    if(maxoffset >= 8) {
      if(offset + 8 < stride) input[offset] = cudaMax<ftype>(input[offset], input[offset + 8]);
    }
    if(maxoffset >= 4) {
      if(offset + 4 < stride) input[offset] = cudaMax<ftype>(input[offset], input[offset + 4]);
    }
    if(maxoffset >= 2) {
      if(offset + 2 < stride) input[offset] = cudaMax<ftype>(input[offset], input[offset + 2]);
    }
    if(maxoffset >= 1) {
      if(offset + 1 < stride) input[offset] = cudaMax<ftype>(input[offset], input[offset + 1]);
    }
  }

  /**
   * @brief Here we find the maximum within a stride. Assumption: One warp does exactly one stride!
   * Reduction via warp reduce. res has the maximum values stored.
   */
  template<int maxoffset>
  static __global__ void findMaxKernelOneWarp(ftype* const res, const ftype* const input, const tensorSize_t stride, const tensorSize_t nStrides) {
    assert(blockDim.x % 32 == 0);

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    const int strideNumber = gid / 32; // same as warp number
    const int withinStrideOffset = gid % 32; // each warp covers up to 32 elements within a stride
    const bool isNotPadded = withinStrideOffset < stride && strideNumber < nStrides;

    extern __shared__ ftype smem[];
    smem[tid] = isNotPadded ? input[strideNumber * stride + withinStrideOffset] : -INFINITY; // TODO: is this memory access pattern bad?
    __syncthreads();

    if(strideNumber >= nStrides) {
      return;
    }

    volatile ftype* const start = smem + (tid / 32) * 32;
    warpMaxReduce<maxoffset>(start, stride, withinStrideOffset);

    if(withinStrideOffset == 0) {
      res[strideNumber] = start[0];
    }
  }

  /**
   * @brief Like findMaxKernelOneWarp. The difference now is that the input size can be much larger. stride is 
   * 64 < stride <= threadsPerBlock. res has the maximum values stored.
   * 
   * In this initial version we assume one kernel per stride, to make matters simple to understand.
   */
  static __global__ void findMaxKernelOneBlock(ftype* const res, const ftype* const input, const tensorSize_t stride) {
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
      smem[tid] = cudaMax<ftype>(input[gid], input[gid + blockDim.x]);
    }
    __syncthreads();

    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] = cudaMax<ftype>(smem[tid], smem[tid + offset]);
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) {
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 32]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 16]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 8]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 4]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 2]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 1]);
    }

    if(tid == 0) { // one block per stride
      res[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Like findMaxKernelOneBlock, but finding partial maximum. This is the case when the stride is too
   * large to fit in one block.
   */
  static __global__ void findMaxKernelLargePass1(ftype* const partialMaxValues, const ftype* const input, 
                                          const tensorSize_t stride, const int blocksPerStride) {
    const int tid = threadIdx.x;
    const int strideIdx = blockIdx.x / blocksPerStride;
    const int blockWithinStride = blockIdx.x % blocksPerStride;

    // block 0 within stride handles elements [0, 2*blockDim.x), block 1 within stride handles elements [2*blockDim.x, 4*blockDim.x), ...
    const int inputBase = strideIdx * stride + blockWithinStride * 2 * blockDim.x; 
      
    extern __shared__ ftype smem[];
    const tensorSize_t localIdx0 = inputBase + tid;
    const tensorSize_t localIdx1 = inputBase + tid + blockDim.x;
      
    // localIdx0 < (strideIdx + 1) * stride <- checks whether thread idx exceeds bounds of this stride; one stride per block at cudaMax<ftype>
    smem[tid] = (localIdx0 < (strideIdx + 1) * stride) ? input[localIdx0] : -INFINITY;
    smem[tid + blockDim.x] = (localIdx1 < (strideIdx + 1) * stride) ? input[localIdx1] : -INFINITY;
    __syncthreads();

    // same reduction as findMaxKernelOneBlock from here
    for(tensorSize_t offset = blockDim.x; offset > 32; offset >>= 1) {
      if(tid < offset){
        smem[tid] = cudaMax<ftype>(smem[tid], smem[tid + offset]);
      } 
      __syncthreads();
    }

    volatile ftype* start = smem;
    if(tid < 32) {
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 32]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 16]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 8]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 4]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 2]);
      start[tid] = cudaMax<ftype>(start[tid], start[tid + 1]);
    }

    if(tid == 0) {
      partialMaxValues[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Self explanatory following findMaxKernelLargePass1. Assumption: All remaining cudaMax<ftype> values do fit into 
   * one single block now -> we launch one block per stride this time.
   */
  static __global__ void findMaxKernelLargePass2(ftype* const maxValues, const ftype* const partialMaxValues, const tensorSize_t blocksPerStride) {
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
      smem[tid] = cudaMax<ftype>(partialMaxValues[gid], partialMaxValues[gid + blockDim.x]);
    }
    __syncthreads();

    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] = cudaMax<ftype>(smem[tid], smem[tid + offset]);
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) {
      if(tid + 32 < blockDim.x) start[tid] = cudaMax<ftype>(start[tid], start[tid + 32]);
      if(tid + 16 < blockDim.x) start[tid] = cudaMax<ftype>(start[tid], start[tid + 16]);
      if(tid + 8  < blockDim.x) start[tid] = cudaMax<ftype>(start[tid], start[tid + 8]);
      if(tid + 4  < blockDim.x) start[tid] = cudaMax<ftype>(start[tid], start[tid + 4]);
      if(tid + 2  < blockDim.x) start[tid] = cudaMax<ftype>(start[tid], start[tid + 2]);
      if(tid + 1  < blockDim.x) start[tid] = cudaMax<ftype>(start[tid], start[tid + 1]);
    }

    if(tid == 0) { // one block per stride
      maxValues[blockIdx.x] = start[0];
    }
  }
}