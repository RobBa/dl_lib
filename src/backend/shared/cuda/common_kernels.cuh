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

#include "utility/global_params.h"
#include "utility/utils.h"

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
    val[0] /= divisor;
  }
}