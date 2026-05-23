/**
 * @file cuda_common.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "cuda_runtime.h"

#include <type_traits>

template<typename T>
struct FtypeWarning {
    static constexpr void check() {}
};

template<>
struct FtypeWarning<double> {
    [[deprecated("ftype=double has serious CUDA performance implications")]]
    static constexpr void check() {}
};

namespace utility {
  void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
}

template <class... T>
constexpr bool always_false = false;

template<typename T>
__device__ __forceinline__ ftype cudaMax(const ftype left, const ftype right) {
  if constexpr (std::is_same_v<T, float>) {
    return fmaxf(a, b);
  } else if(std::is_same_v<T, double>) {
      return fmax(a, b);
  }
  else {
    always_false<T, "Unexpected value for ftype encountered">;
  }
}

/**
  * @brief Single sigmoid computation.
  */
__device__ __forceinline__ ftype cudaSigmoid(ftype x) {
  ftype z = expf(-fabsf(x));
  ftype s = 1.0f / (1.0f + z);
  return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
}

#define cudaErrchk(ans) { utility::gpuAssert((ans), __FILE__, __LINE__); }

namespace cuda_impl {
  struct DeviceProperties final {
    private:
      int threadsPerBlock;
      int warpSize;

      static const DeviceProperties& get() {
        static DeviceProperties instance;
        return instance;
      }

      DeviceProperties() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        threadsPerBlock = prop.maxThreadsPerBlock;
        warpSize = prop.warpSize;
      }

    public:
      DeviceProperties(const DeviceProperties&) = delete;
      DeviceProperties& operator=(const DeviceProperties&) = delete;
      
      DeviceProperties(DeviceProperties&&) = delete;
      DeviceProperties& operator=(DeviceProperties&&) = delete;
      
      ~DeviceProperties() = default;

      static int getThreadsPerBlock() { return get().threadsPerBlock; }
      static int getWarpSize() { return get().warpSize; }
  };
}
