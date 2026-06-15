/**
 * @file cuda_common.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif

#include "cuda_common.cuh"

#include <iostream>

namespace {
  const char* curandGetErrorString(curandStatus_t status) {
    switch (status) {
      case CURAND_STATUS_SUCCESS: return "SUCCESS";
      case CURAND_STATUS_VERSION_MISMATCH: return "VERSION_MISMATCH";
      case CURAND_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
      case CURAND_STATUS_ALLOCATION_FAILED: return "ALLOCATION_FAILED";
      case CURAND_STATUS_TYPE_ERROR: return "TYPE_ERROR";
      case CURAND_STATUS_OUT_OF_RANGE: return "OUT_OF_RANGE";
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "LENGTH_NOT_MULTIPLE";
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "DOUBLE_PRECISION_REQUIRED";
      case CURAND_STATUS_LAUNCH_FAILURE: return "LAUNCH_FAILURE";
      case CURAND_STATUS_PREEXISTING_FAILURE: return "PREEXISTING_FAILURE";
      case CURAND_STATUS_INITIALIZATION_FAILED: return "INITIALIZATION_FAILED";
      case CURAND_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
      case CURAND_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
      default: return "UNKNOWN";
    }
  }

  const char* cudaMemoryTypeToString(cudaMemoryType type) {
    switch(type) {
      case cudaMemoryTypeUnregistered: return "Unregistered host";
      case cudaMemoryTypeHost:         return "Pinned host";
      case cudaMemoryTypeDevice:       return "Device";
      case cudaMemoryTypeManaged:      return "Managed";
      default:                         return "Unknown";
    }
  }
}

namespace utility {
  void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) 
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
        if (abort) exit(code);
    }
  }

  void cuRandAssert(curandStatus_t status, const char *file, int line, bool abort) {
    if (status != CURAND_STATUS_SUCCESS) { \
      std::cerr << "CuRandAssert: " << curandGetErrorString(status) << " " << file << " " << line << "\n";
      if (abort) exit(status);
    }
  }

  void printPtrProperties(ftype* ptr) {
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        printf("cudaPointerGetAttributes failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("type=%s, device=%d\n", cudaMemoryTypeToString(attrs.type), attrs.device);
    }
  }
}