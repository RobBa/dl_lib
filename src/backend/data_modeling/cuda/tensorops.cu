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
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "tensorops.cuh"
#include "utility/cuda/cuda_common.cuh"

#include "cuda_runtime.h"

#include "data_modeling/tensor.h"

namespace{
  __global__ void elementwiseaddKernel(ftype* res, const ftype* const left, const ftype* const right, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    res[gid] = left[gid] + right[gid];
  }

  __global__ void elementwisemulKernel(ftype* res, const ftype* const left, const ftype* const right, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    res[gid] = left[gid] * right[gid];
  }

  __global__ void scalaraddKernel(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size){
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    res[gid] = left[gid] + scalar;
  }

  __global__ void scalarmulKernel(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    res[gid] = left[gid] + scalar;
  }

  __global__ void createLinearCopyKernel(
      float* dst, const float* const src,
      const tensorSize_t* const strides,       // original strides
      const tensorSize_t* const contiguousStrides, // new linear strides
      int ndim, tensorSize_t size)
  {
      tensorSize_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x;
      if (flatIdx >= size) return;

      tensorSize_t remainder = flatIdx;
      tensorSize_t srcOffset = 0;
      for (int i = 0; i < ndim; ++i) {
          tensorSize_t coord = remainder / contiguousStrides[i];
          remainder %= contiguousStrides[i];
          srcOffset += coord * strides[i];
      }
      dst[flatIdx] = src[srcOffset];
  }
}

namespace cuda {
  void scalaradd(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalaraddKernel<<<blocksPerGrid, threadsPerBlock>>>(res, left, scalar, size);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void scalarmul(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalarmulKernel<<<blocksPerGrid, threadsPerBlock>>>(res, left, scalar, size);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void elementwiseadd(ftype* res, const ftype* const left, const ftype* const right, tensorSize_t size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    elementwiseaddKernel<<<blocksPerGrid, threadsPerBlock>>>(res, left, right, size);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void elementwisemul(ftype* res, const ftype* const left, const ftype* const right, tensorSize_t size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    elementwisemulKernel<<<blocksPerGrid, threadsPerBlock>>>(res, left, right, size);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void matmul(ftype* res, const ftype* const left, const ftype* const right) {
    // TODO
  }

  ftype get(const ftype* const t, tensorSize_t idx) {
    ftype res;
    cudaErrchk(cudaMemcpy(&res, t+idx, sizeof(ftype), cudaMemcpyDeviceToHost));
    return res;
  }

  ftype set(ftype value, const ftype* t, tensorSize_t idx) {
    cudaErrchk(cudaMemcpy((void*)t+idx, &value, sizeof(ftype), cudaMemcpyHostToDevice));
  }

  void createLinearCopy(Tensor& res, const Tensor& src) {
    assert(res.getSize()==src.getSize());

    ftype* dst = res.getData()
    const ftype* const srcData = src.getData();

    auto oldStrides = src.getDims().getCreationStrides().data();
    auto newStrides = src.getDims().getStrides().data();

    int threadsPerBlock = 256;
    int blocksPerGrid = (nBytes + threadsPerBlock - 1) / threadsPerBlock;

    cudaErrchk(createLinearCopyKernel<<<threadsPerBlock, blocksPerGrid>>>(
      dst, srcData, oldStrides, newStrides, dims.nDims(), nBytes));

    cudaErrchk(cudaDeviceSynchronize());
  }
}
