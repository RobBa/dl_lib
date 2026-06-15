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
#include "curand.h"

#include "shared/global_params.h"

#include <type_traits>

namespace utility {
  struct DeviceProperties;

  void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
  void cuRandAssert(curandStatus_t status, const char *file, int line, bool abort=false);
  
  void printPtrProperties(ftype* ptr);
}

#define cudaErrchk(ans) { utility::gpuAssert((ans), __FILE__, __LINE__); }
#define cuRandErrchk(ans) { utility::cuRandAssert((ans), __FILE__, __LINE__); }

#define ASSERT_DEVICE_PTR(ptr) { \
  cudaPointerAttributes _attrs{}; \
  cudaPointerGetAttributes(&_attrs, ptr); \
  assert(_attrs.type == cudaMemoryTypeDevice && "Expected device pointer"); \
}

#define ASSERT_HOST_PTR(ptr) { \
  cudaPointerAttributes _attrs{}; \
  cudaPointerGetAttributes(&_attrs, ptr); \
  assert(_attrs.type == cudaMemoryTypeUnregistered && "Expected host pointer"); \
}

namespace utility {
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
