/**
 * @file initializers.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "initializers.cuh"

#include "shared/cuda/common_kernels.cuh"
#include "utility/cuda/cuda_common.cuh"

namespace cuda_impl {
  void scaleArr(ftype* const data, const ftype scale, const ftype shift, const tensorSize_t size) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    cuda_impl::scalePlusOffsetKernel<<<blocks, threadsPerBlock>>>(data, scale, shift, size);
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }
}