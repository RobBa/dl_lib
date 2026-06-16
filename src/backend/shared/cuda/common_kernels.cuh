/**
 * @file common_kernels.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
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

#include "shared/global_params.h"
#include "utility/utils.h"

#include <cassert>

namespace cuda_impl {
  template<typename T>
  __device__ __forceinline__ ftype cudaMax(const ftype a, const ftype b) {
    if constexpr (std::is_same_v<T, float>) {
      return fmaxf(a, b);
    } 
    else if constexpr (std::is_same_v<T, double>) {
        return fmax(a, b);
    }
    else {
      static_assert(always_false<T>, "Unexpected value for ftype encountered");
    }
  }

  /**
    * @brief Single sigmoid computation.
    */
   template<typename T>
  __device__ __forceinline__ ftype cudaSigmoid(const ftype x) {
    if constexpr (std::is_same_v<T, float>) {
      ftype z = expf(-fabsf(x));
      ftype s = 1.0f / (1.0f + z);
      return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
    } 
    else if constexpr (std::is_same_v<T, double>) {
      ftype z = exp(-abs(x));
      ftype s = 1.0 / (1.0 + z);
      return (x >= 0.0) ? s : z * s; // x < 0 => e^x/(e^x+1) 
    }
    else {
      static_assert(always_false<T>, "Unexpected value for ftype encountered");
    }
  }

  template<typename T>
  __device__ __forceinline__ ftype cudaSqrt(const ftype x) {
    if constexpr (std::is_same_v<T, float>) {
      return sqrtf(x);
    } 
    else if constexpr (std::is_same_v<T, double>) {
      return sqrt(x);
    }
    else {
      static_assert(always_false<T>, "Unexpected value for ftype encountered");
    }
  }

  /**
   * @brief For single normalization, e.g. when normalizing with batch-size.
   */
  static __global__ void divideScalarKernel(ftype* const val, const ftype divisor) {
    assert(blockDim.x == 1 && gridDim.x == 1);
    val[0] /= divisor;
  }

  /**
   * @brief Multiplication with scalar + an offset.
   */
  static __global__ void scalePlusOffsetKernel(ftype* const data, const ftype scale, const ftype shift, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) {
      return;
    }

    data[gid] = data[gid] * scale + shift;
  }

  /**
   * @brief A simple small reduction kernel that sums over an array, returning the 
   * sum in the inOut parameter. Assumes that input and output fit in one block! 
   * Also, assumes that blockDim.x < inputSize, since each thread sums up 
   * to two elements. input and output are allowed to be the same array.
   * 
   * 
   * @param output The output where the sum over input will be written into. 
   * Result written to ouput[ix]
   * @param idx The idx to write to.
   * @param inputSize Size of the input for boundary checks.
   */
  static __global__ void sumReduceKernel(ftype* const output, const ftype* const input, const int idx, const tensorSize_t inputSize) {
    assert(gridDim.x == 1);

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ ftype smem[]; 
    const ftype x1 = gid < inputSize ? input[gid] : 0.0f;
    const ftype x2 = gid + blockDim.x < inputSize ? input[gid + blockDim.x] : 0.0f;
    smem[tid] = x1 + x2;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 16; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid < 32) {
      assert(blockDim.x >= 32);
      
      ftype sum = smem[tid];
      for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if(tid == 0) {
        output[idx] = sum;
      }
    }
  }
}