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

#include "utility/utils.h"
#include "utility/cuda/cuda_common.cuh"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

using namespace std;

namespace {
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
   * @param leftRows n rows of left. !All rows and column params refer to the values as they are in physical 
   * memory, regardless of transpose!
   * @param rightRows n rows of right. See also leftRows.
   * @param leftCols n columns of left matrix. See also leftRows.
   * @param rightCols n columns of right matrix. See also leftRows.
   * @param resSize Size of the resulting 2D matrix.
   */
  template<bool transposeLeft, bool transposeRight>
  __global__ void matMul2DKernel(ftype* const res, 
                                 const ftype* const left, const ftype* const right,
                                 const tensorDim_t leftRows, const tensorDim_t rightRows, 
                                 const tensorDim_t leftCols, const tensorDim_t rightCols, 
                                 const tensorSize_t resSize)
  {
    const int tid = threadIdx.x;
    const int gid = blockDim.x * blockIdx.x + tid;
    if(gid >= resSize) 
      return;

    // 48KB of shared mem -> ~12K floats of 4 bytes -> ~10 blocks per SM of shared memory is limit
    //extern __shared__ ftype smem[];
    //smem[tid] = 0.0f;
    //__syncthreads();

    const int K = transposeLeft ? leftRows :  leftCols;
    const int N = transposeRight ? rightRows : rightCols;

    const int i = gid / N;
    const int j = gid % N;

    // C[i, j] = sum_{k=0}^{leftCols} A[i, k] * B[k, j]
    ftype cij = 0;
    for (int k = 0; k < K; k++) {
      //smem[tid] += left[leftBase + k] * right[k * rightCols + resCol];

      int leftIdx = transposeLeft ? k * leftCols + i : i * leftCols + k;
      int rightIdx = transposeRight ? j * rightCols + k : k * rightCols + j;
            
      cij += left[leftIdx] * right[rightIdx];
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
  __global__ void createContiguousCopyKernel(ftype* const dst, const ftype* const src, const tensorSize_t* const strides,
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

  /**
   * @brief Copies the elements of the given slices from src to res. Assumes that the dimension to slice over
   * is the first one currently!
   * 
   * @param idx The indices of the dimension in src.
   */
  __global__ void getSliceKernel(ftype* res, const ftype* const src, const tensorDim_t* const idx,
                                 const tensorSize_t sizeOfDim, const tensorSize_t resSize) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= resSize) {
      return;
    }

    const int strideNumber = gid / sizeOfDim;
    const int withinStrideIdx = gid % sizeOfDim;

    const int srcIdx = idx[strideNumber] * sizeOfDim + withinStrideIdx;
    const int resIdx = strideNumber * sizeOfDim + withinStrideIdx;
    
    res[resIdx] = src[srcIdx];
  }
}

namespace cuda_impl {
  void scalaradd(Tensor& res, const Tensor& src, ftype scalar) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (src.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    scalaraddKernel<<<blocks, threadsPerBlock>>>(res.getData(), src.getData(), scalar, src.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void scalarmul(Tensor& res, const Tensor& src, ftype scalar) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (src.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    scalarmulKernel<<<blocks, threadsPerBlock>>>(res.getData(), src.getData(), scalar, src.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void broadcastadd(Tensor& res, const Tensor& matrix, const Tensor& vec) {
    const auto size = res.getSize();

    constexpr int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    broadcastaddKernel<<<blocks, threadsPerBlock>>>(
      res.getData(), matrix.getData(), vec.getData(), vec.getDims()[0], matrix.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void elementwiseadd(Tensor& res, const Tensor& left, const Tensor& right) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (left.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseaddKernel<<<blocks, threadsPerBlock>>>(res.getData(), left.getData(), right.getData(), left.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void elementwisemul(Tensor& res, const Tensor& left, const Tensor& right) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (left.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    elementwisemulKernel<<<blocks, threadsPerBlock>>>(res.getData(), left.getData(), right.getData(), left.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void matmul(Tensor& res, const Tensor& left, const Tensor& right, const bool transposeLeft, const bool transposeRight) {
    constexpr int threadsPerBlock = 256;
    
    // sizes of the 2D matrices respectively
    const tensorSize_t leftSize = left.getDims().get(-1) * left.getDims().get(-2); 
    const tensorSize_t rightSize = right.getDims().get(-1) * right.getDims().get(-2);
    const tensorSize_t resSize = res.getDims().get(-2) * res.getDims().get(-1);

    const int blocks = (resSize + threadsPerBlock - 1) / threadsPerBlock;

    tensorSize_t leftOffset = 0;
    tensorSize_t rightOffset = 0;
    tensorSize_t resOffset = 0;

    while(leftOffset < left.getSize()){
      //const auto smemSize = min(resSize, threadsPerBlock) * sizeof(ftype);
      //matMul2DKernel<<<blocks, threadsPerBlock, smemSize>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset,
      if(!(transposeLeft || transposeRight)) {
        matMul2DKernel<false, false><<<blocks, threadsPerBlock>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset, 
                                       left.getDims().get(-2), right.getDims().get(-2), left.getDims().get(-1), right.getDims().get(-1), resSize);
      }
      else if(transposeLeft && transposeRight) [[unlikely]] {
        matMul2DKernel<true, true><<<blocks, threadsPerBlock>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset, 
                                     left.getDims().get(-2), right.getDims().get(-2), left.getDims().get(-1), right.getDims().get(-1), resSize);
      }
      else if(transposeLeft) {
        matMul2DKernel<true, false><<<blocks, threadsPerBlock>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset, 
                                      left.getDims().get(-2), right.getDims().get(-2), left.getDims().get(-1), right.getDims().get(-1), resSize);
      }
      else if(transposeRight) {
        matMul2DKernel<false, true><<<blocks, threadsPerBlock>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset, 
                                      left.getDims().get(-2), right.getDims().get(-2), left.getDims().get(-1), right.getDims().get(-1), resSize);
      }

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
    assert(res.getSize() == src.getSize());

    const auto size = src.getSize();
    constexpr int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    createContiguousCopyKernel<<<blocks, threadsPerBlock>>>(
      res.getData(), src.getData(), src.getDims().getStrides().data(), src.getDims().data(), src.getDims().nDims(), size);
    cudaErrchk(cudaDeviceSynchronize());
  }

  void getSlice(Tensor& res, const Tensor& src, span<const tensorDim_t> idx) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (res.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    tensorDim_t* idx_d;
    cudaErrchk(cudaMalloc(&idx_d, idx.size() * sizeof(tensorDim_t)));
    const auto sizeOfDim = res.getDims().getStride(0);

    cudaErrchk(cudaMalloc(&idx_d, idx.size() * sizeof(tensorDim_t)));
    cudaErrchk(cudaMemcpy(idx_d, idx.data(), idx.size() * sizeof(tensorDim_t), cudaMemcpyHostToDevice));

    getSliceKernel<<<blocks, threadsPerBlock>>>(res.getData(), src.getData(), idx_d, sizeOfDim, res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaFree(idx_d));
  }
}
