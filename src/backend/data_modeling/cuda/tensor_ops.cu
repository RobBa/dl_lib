/**
 * @file tensor_ops.cu
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

#include "tensor_ops.cuh"
#include "utility/cuda/cuda_common.cuh"
#include "utility/utils.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

namespace{
  /**
   * @brief Kernel for simple elementwise addition.
   */
  __global__ void elementwiseaddKernel(ftype* const res, const ftype* const left, const ftype* const right, const tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;
    
    res[gid] = left[gid] + right[gid];
  }

  /**
   * @brief Kernel for broadcasted addition, e.g. when adding a bias to a matrix.
   */
  __global__ void broadcastaddKernel(ftype* const res, const ftype* const matrix, const ftype* const vec, 
                                     const tensorSize_t vectorSize, const tensorSize_t matrixSize) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=matrixSize)
        return;

    const int vectorIdx = gid % vectorSize;
    res[gid] = matrix[gid] + vec[vectorIdx];
  }

  /**
   * @brief Kernel for simple elementwise multiplication.
   */
  __global__ void elementwisemulKernel(ftype* const res, const ftype* const left, const ftype* const right, const tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;
    
    res[gid] = left[gid] * right[gid];
  }

  /**
   * @brief Kernel for scalar addition.
   */
  __global__ void scalaraddKernel(ftype* const res, const ftype* const left, ftype scalar, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;
    
    res[gid] = left[gid] + scalar;
  }

  /**
   * @brief Kernel for scalar multiplication.
   */
  __global__ void scalarmulKernel(ftype* const res, const ftype* const left, ftype scalar, tensorSize_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= size)
      return;
    
    res[gid] = left[gid] * scalar;
  }

  /**
   * @brief 2D matMul operation. Assumes that res is where result is written to, and res is zero-indexed.
   * For higher dimensionalities the caller is responsible for proper offset computation!!! The same goes
   * for left and right, which are also assumed to be 2D matrices, and offsets have to be computed by caller.
   * 
   * @param res The result matrix
   * @param left Left matrix to be multiplied
   * @param right Right matrix to multiply with
   * @param leftRows n rows of 
   * @param leftCols n columns of left matrix; leftCols == rightRows => rightRows not handed as param
   * @param rightCols n columns of right matrix.
   * @param resSize Size of the resulting 2D matrix.
   */
  __global__ void matMul2DKernel(ftype* const res, const ftype* const left, const ftype* const right,
                               const tensorDim_t leftRows, const tensorDim_t leftCols,
                               const tensorDim_t rightCols, const tensorSize_t resSize)
  {
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= resSize) 
      return;
    //const int tid = threadIdx.x;

    // 48KB of shared mem -> ~12K floats of 4 bytes -> ~10 blocks per SM of shared memory is limit
    //extern __shared__ ftype smem[];
    //smem[tid] = 0.0f;
    //__syncthreads();

    const int resCol = gid % rightCols;
    const int resRow = gid / rightCols;
    const int leftBase = resRow * leftCols;

    // C[i, j] = sum_{k=0}^{leftCols} A[i, k] * B[k, j]
    ftype cij = 0;
    for(int k = 0; k < leftCols; k++) {
      //smem[tid] += left[leftBase + k] * right[k * rightCols + resCol];
      cij += left[leftBase + k] * right[k * rightCols + resCol];
    }

    res[gid] = cij; // smem[tid];
  }

  /**
   * @brief Strides in an outer size larger than one block. We use one thread per stride.
   */
  __global__ void sumOverDimsKernel(ftype* const res, const ftype* const input, tensorSize_t stride,
                                    const tensorDim_t srcDimSize, const tensorSize_t size) {
    const tensorSize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    } 

    const tensorSize_t outerIdx = gid / stride;
    const tensorSize_t batchOffset = outerIdx * stride * srcDimSize;

    const int withinStrideIdx = gid % stride;

    ftype sum = 0;
    for(int k = 0; k < srcDimSize; k++) {
      sum += input[batchOffset + k * stride + withinStrideIdx];
    }

    res[gid] = sum;
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
    const tensorSize_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(flatIdx >= size) 
      return;

    tensorSize_t remainder = flatIdx;
    tensorSize_t srcOffset = 0;
    for (int i = ndims - 1; i >= 0; i--) {
      tensorSize_t coord = remainder % dims[i];
      srcOffset += coord * strides[i];
      remainder /= dims[i];
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
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (res.getSize() + threadsPerBlock - 1) / threadsPerBlock;
    
    // sizes of the 2D matrices respectively
    const tensorSize_t leftSize = left.getDims().get(-1) * left.getDims().get(-2); 
    const tensorSize_t rightSize = right.getDims().get(-1) * right.getDims().get(-2);
    const tensorSize_t resSize = left.getDims().get(-2) * right.getDims().get(-1);

    tensorSize_t leftOffset = 0;
    tensorSize_t rightOffset = 0;
    tensorSize_t resOffset = 0;

    while(leftOffset < left.getSize()){
      //const auto smemSize = min(resSize, threadsPerBlock) * sizeof(ftype);
      //matMul2DKernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset, 
      matMul2DKernel<<<blocksPerGrid, threadsPerBlock>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset, 
                                                         left.getDims().get(-2), left.getDims().get(-1), right.getDims().get(-1), resSize);

      leftOffset += leftSize;
      rightOffset += rightSize;
      resOffset += resSize;
    }
    
    cudaErrchk(cudaDeviceSynchronize());
  }

  void sumOverDims(Tensor& res, const Tensor& input, tensorDim_t dim) {
    tensorSize_t stride = 1;
    for(tensorDim_t i = dim + 1; i < input.getDims().nDims(); i++){
      stride *= input.getDims()[i];
    }

    constexpr int threadsPerBlock = 256;
    const int blocks = (res.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    sumOverDimsKernel<<<blocks, threadsPerBlock>>>(res.getData(), input.getData(), stride, input.getDims()[dim], res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
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
