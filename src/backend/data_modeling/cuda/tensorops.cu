/**
 * @file tensorops.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-23
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "data_modeling/tensor.h"

#include "tensorops.cuh"
#include "utility/cuda/cuda_common.cuh"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

namespace{
  __global__ void elementwiseaddKernel(ftype* res, const ftype* const left, const ftype* const right, const tensorSize_t size)
  {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=size)
      return;
    
    res[gid] = left[gid] + right[gid];
  }

  __global__ void broadcastaddKernel(ftype* res, const ftype* const matrix, const ftype* const vec, 
                                     const tensorSize_t vectorSize, const tensorSize_t matrixSize)
  {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=matrixSize)
        return;

    const int vectorIdx = gid % vectorSize;
    res[gid] = matrix[gid] + vec[vectorIdx];
  }

  __global__ void elementwisemulKernel(ftype* res, const ftype* const left, const ftype* const right, const tensorSize_t size)
  {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=size)
      return;
    
    res[gid] = left[gid] * right[gid];
  }

  __global__ void scalaraddKernel(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size)
  {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=size)
      return;
    
    res[gid] = left[gid] + scalar;
  }

  __global__ void scalarmulKernel(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size)
  {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=size)
      return;
    
    res[gid] = left[gid] * scalar;
  }

  __global__ void matMulKernel(ftype* res, const ftype* const left, const ftype* const right, 
                               const tensorDim_t leftRows, const tensorDim_t leftCols, tensorDim_t rightRows, tensorDim_t rightCols)
  {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    //if()
  }

  /**
   * @brief Create a contiguous copy of src and copy it into dst. Used for reshaping, transposing, etc.
   * 
   * @param strides The original strides of src. With these strides src would be contiguous.
   * @param contiguousStrides The new, contiguous strides of dst. We rearrange to match those in memory.
   * @param ndim Number of dimension of both src and dst.
   * @param size Total size of both src and dst.
   */
  __global__ void createContiguousCopyKernel(ftype* dst, const ftype* const src, const tensorSize_t* const strides,
                                             const tensorDim_t* const dims, const int ndims, const tensorSize_t size)
  {
    tensorSize_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(flatIdx >= size) 
      return;

    tensorSize_t remainder = flatIdx;
    tensorSize_t srcOffset = 0;
    for (int i = ndims - 1; i >= 0; i--) {
      tensorSize_t coord = remainder % dims[i];
      remainder /= dims[i];
      srcOffset += coord * strides[i];
    }
    dst[flatIdx] = src[srcOffset];
  }
}

namespace cuda_impl {
  void scalaradd(Tensor& res, const Tensor& src, ftype scalar) {
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (src.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    scalaraddKernel<<<blocksPerGrid, threadsPerBlock>>>(res.getData(), src.getData(), scalar, src.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void scalarmul(Tensor& res, const Tensor& src, ftype scalar) {
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (src.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    scalarmulKernel<<<blocksPerGrid, threadsPerBlock>>>(res.getData(), src.getData(), scalar, src.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  // TODO: fix this one
  void broadcastadd(Tensor& res, const Tensor& matrix, const Tensor& vec) {
    const auto size = res.getSize();

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    broadcastaddKernel<<<blocksPerGrid, threadsPerBlock>>>(
      res.getData(), matrix.getData(), vec.getData(), vec.getDims()[0], matrix.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void elementwiseadd(Tensor& res, const Tensor& left, const Tensor& right) {
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (left.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseaddKernel<<<blocksPerGrid, threadsPerBlock>>>(res.getData(), left.getData(), right.getData(), left.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void elementwisemul(Tensor& res, const Tensor& left, const Tensor& right) {
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (left.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    elementwisemulKernel<<<blocksPerGrid, threadsPerBlock>>>(res.getData(), left.getData(), right.getData(), left.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void matmul(Tensor& res, const Tensor& left, const Tensor& right) {
/*     constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(res, left, right, size);
    cudaErrchk(cudaDeviceSynchronize()); */
  }

  void scalarFill(Tensor& t, ftype value) {
    ftype* ptr = t.getData();
    thrust::fill(thrust::device_pointer_cast(ptr),
                 thrust::device_pointer_cast(ptr + t.getSize()), value);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void createContiguousCopy(Tensor& res, const Tensor& src) {
    assert(res.getSize()==src.getSize());

    const auto size = src.getSize();
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    createContiguousCopyKernel<<<blocksPerGrid, threadsPerBlock>>>(
      res.getData(), src.getData(), src.getDims().getStrides().data(), src.getDims().data(), src.getDims().nDims(), size);
    cudaErrchk(cudaDeviceSynchronize());
  }
}
