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

#include "tensor_ops.cuh"
#include "data_modeling/tensor.h"

#include "shared/memory_pool.h"

#include "utility/utils.h"
#include "utility/cuda/cuda_common.cuh"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#ifdef CUDA_BACKEND_CUBLAS
#include <cublas_v2.h>
#endif

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

#ifdef CUDA_BACKEND_CUBLAS
  cublasHandle_t& cublasHandle() {
    static cublasHandle_t handle = [] {
      cublasHandle_t h;
      cublasCreate(&h);
      return h;
    }();
    return handle;
  }

  template<typename T>
  void cublasGemmT(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                   int m, int n, int k, const T* alpha,
                   const T* A, int lda, const T* B, int ldb,
                   const T* beta, T* C, int ldc);

  template<> void cublasGemmT<ftype>(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                   int m, int n, int k, const ftype* alpha,
                   const ftype* A, int lda, const ftype* B, int ldb,
                   const ftype* beta, ftype* C, int ldc) {
    cublasSgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  template<typename T>
  void cublasGemmStridedBatchedT(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                   int m, int n, int k, const T* alpha,
                   const T* A, int lda, long long strideA,
                   const T* B, int ldb, long long strideB,
                   const T* beta, T* C, int ldc, long long strideC, int batchCount);

  template<> void cublasGemmStridedBatchedT<ftype>(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                   int m, int n, int k, const ftype* alpha,
                   const ftype* A, int lda, long long strideA,
                   const ftype* B, int ldb, long long strideB,
                   const ftype* beta, ftype* C, int ldc, long long strideC, int batchCount) {
    cublasSgemmStridedBatched(h, opA, opB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  }
#else

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
  template<int tileSize, bool transposeLeft, bool transposeRight>
  __global__ void matMul2DKernel(ftype* const res, 
                                 const ftype* const left, const ftype* const right,
                                 const tensorDim_t leftRows, const tensorDim_t leftCols, 
                                 const tensorDim_t rightRows, const tensorDim_t rightCols,
                                 const tensorDim_t resRows, const tensorDim_t resCols,
                                 const tensorSize_t leftSize, const tensorSize_t rightSize, const tensorSize_t resSize)
  {
    __shared__ ftype smemA[tileSize];
    __shared__ ftype smemB[tileSize];

    const tensorSize_t K = transposeLeft ? leftRows : leftCols;
    const tensorSize_t N = transposeRight ? rightRows : rightCols;

    const tensorSize_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const tensorSize_t j = blockIdx.x * blockDim.x + threadIdx.x;

    const tensorSize_t leftOffset = blockIdx.z * leftSize;
    const tensorSize_t rightOffset = blockIdx.z * rightSize;
    const tensorSize_t resOffset = blockIdx.z * resSize;

    ftype cij = 0.0f;
    for (tensorSize_t k = 0; k < K; k += blockDim.x) {
      // load tile into smem
      const tensorSize_t leftIdx = transposeLeft ? (k + threadIdx.x) * leftCols + i : i * leftCols + (k + threadIdx.x);
      const tensorSize_t rightIdx = transposeRight ? j * rightCols + (k + threadIdx.y) : (k + threadIdx.y) * rightCols + j;

      smemA[threadIdx.y * blockDim.x + threadIdx.x] = (i < resRows && (k + threadIdx.x) < K) ? left[leftOffset + leftIdx] : 0.0f;
      smemB[threadIdx.y * blockDim.x + threadIdx.x] = ((k + threadIdx.y) < K && j < resCols) ? right[rightOffset + rightIdx] : 0.0f;
      __syncthreads();

      for (int l = 0; l < blockDim.x; l++) {
        cij += smemA[threadIdx.y * blockDim.x + l] * smemB[l * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    if (i < resRows && j < resCols) {
      res[resOffset + i * N + j] = cij;
    }
  }

#endif

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

    ftype sum = 0.0f;
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
   * is the first one currently! I.e. we can only slice over dim0 right now!
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
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void scalarmul(Tensor& res, const Tensor& src, ftype scalar) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (src.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    scalarmulKernel<<<blocks, threadsPerBlock>>>(res.getData(), src.getData(), scalar, src.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void broadcastadd(Tensor& res, const Tensor& matrix, const Tensor& vec) {
    const auto size = res.getSize();

    constexpr int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    broadcastaddKernel<<<blocks, threadsPerBlock>>>(
      res.getData(), matrix.getData(), vec.getData(), vec.getDims()[0], matrix.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void elementwiseadd(Tensor& res, const Tensor& left, const Tensor& right) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (left.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseaddKernel<<<blocks, threadsPerBlock>>>(res.getData(), left.getData(), right.getData(), left.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void elementwisemul(Tensor& res, const Tensor& left, const Tensor& right) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (left.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    elementwisemulKernel<<<blocks, threadsPerBlock>>>(res.getData(), left.getData(), right.getData(), left.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void matmul(Tensor& res, const Tensor& left, const Tensor& right, const bool transposeLeft, const bool transposeRight) {
    const tensorSize_t leftSize  = left.getDims().get(-2)  * left.getDims().get(-1);
    const tensorSize_t rightSize = right.getDims().get(-2) * right.getDims().get(-1);
    const tensorSize_t resSize   = res.getDims().get(-2)   * res.getDims().get(-1);

#ifdef CUDA_BACKEND_CUBLAS
    //we need op_A(right^T) = op_R(right)^T,
    const cublasOperation_t opA = transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opB = transposeLeft  ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int m    = static_cast<int>(res.getDims().get(-1));   // resCols
    const int n    = static_cast<int>(res.getDims().get(-2));   // resRows
    const int k    = static_cast<int>(transposeLeft ? left.getDims().get(-2) : left.getDims().get(-1));
    const int lda  = static_cast<int>(right.getDims().get(-1)); // rightCols
    const int ldb  = static_cast<int>(left.getDims().get(-1));  // leftCols
    const int ldc  = m;

    const ftype alpha = static_cast<ftype>(1.0);
    const ftype beta  = static_cast<ftype>(0.0);

    const int batchCount = static_cast<int>(res.getSize() / resSize);

    if(batchCount == 1) {
      cublasGemmT<ftype>(cublasHandle(), opA, opB, m, n, k,
                         &alpha, right.getData(), lda, left.getData(), ldb,
                         &beta,  res.getData(),   ldc);
    } else {
      cublasGemmStridedBatchedT<ftype>(cublasHandle(), opA, opB, m, n, k,
                         &alpha,
                         right.getData(), lda, static_cast<long long>(rightSize),
                         left.getData(),  ldb, static_cast<long long>(leftSize),
                         &beta,
                         res.getData(),   ldc, static_cast<long long>(resSize),
                         batchCount);
    }

#else
    constexpr int MATMUL_TILESIZE = 16; // choose 16 (threadsPerBlock=256) or 32 (threadsPerBlock=1024)
    constexpr dim3 threadsPerBlock(MATMUL_TILESIZE, MATMUL_TILESIZE);

    const tensorSize_t nMultiplications = res.getSize() / resSize;

    // x = horizontal, y = vertical
    const int blocksX = (res.getDims().get(-1) + MATMUL_TILESIZE - 1) / MATMUL_TILESIZE;
    const int blocksY = (res.getDims().get(-2) + MATMUL_TILESIZE - 1) / MATMUL_TILESIZE;
    dim3 numBlocks(blocksX, blocksY, nMultiplications);

    //const auto smemSize = min(resSize, threadsPerBlock) * sizeof(ftype);
    //matMul2DKernel<<<blocks, threadsPerBlock, smemSize>>>(res.getData() + resOffset, left.getData() + leftOffset, right.getData() + rightOffset,
    if(!(transposeLeft || transposeRight)) {
      matMul2DKernel<MATMUL_TILESIZE * MATMUL_TILESIZE, false, false><<<numBlocks, threadsPerBlock>>>(
                                                      res.getData(), left.getData(), right.getData(),
                                                      left.getDims().get(-2), left.getDims().get(-1),
                                                      right.getDims().get(-2), right.getDims().get(-1),
                                                      res.getDims().get(-2), res.getDims().get(-1),
                                                      leftSize, rightSize, resSize);
    }
    else if(transposeLeft && transposeRight) [[unlikely]] {
      matMul2DKernel<MATMUL_TILESIZE * MATMUL_TILESIZE, true, true><<<numBlocks, threadsPerBlock>>>(
                                                    res.getData(), left.getData(), right.getData(),
                                                    left.getDims().get(-2), left.getDims().get(-1),
                                                    right.getDims().get(-2), right.getDims().get(-1),
                                                    res.getDims().get(-2), res.getDims().get(-1),
                                                    leftSize, rightSize, resSize);
    }
    else if(transposeLeft) {
      matMul2DKernel<MATMUL_TILESIZE * MATMUL_TILESIZE, true, false><<<numBlocks, threadsPerBlock>>>(
                                                     res.getData(), left.getData(), right.getData(),
                                                     left.getDims().get(-2), left.getDims().get(-1),
                                                     right.getDims().get(-2), right.getDims().get(-1),
                                                     res.getDims().get(-2), res.getDims().get(-1),
                                                     leftSize, rightSize, resSize);
    }
    else if(transposeRight) {
      matMul2DKernel<MATMUL_TILESIZE * MATMUL_TILESIZE, false, true><<<numBlocks, threadsPerBlock>>>(
                                                     res.getData(), left.getData(), right.getData(),
                                                     left.getDims().get(-2), left.getDims().get(-1),
                                                     right.getDims().get(-2), right.getDims().get(-1),
                                                     res.getDims().get(-2), res.getDims().get(-1),
                                                     leftSize, rightSize, resSize);
    }
#endif

    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void sumOverDims(Tensor& res, const Tensor& input, tensorDim_t dim) {
    tensorSize_t stride = 1;
    for(tensorDim_t i = dim + 1; i < input.getDims().nDims(); i++){
      stride *= input.getDims()[i];
    }

    constexpr int threadsPerBlock = 256;
    const int blocks = (res.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    sumOverDimsKernel<<<blocks, threadsPerBlock>>>(res.getData(), input.getData(), stride, input.getDims()[dim], res.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void scalarFill(Tensor& t, ftype value) {
    ftype* ptr = t.getData();
    thrust::fill(thrust::device_pointer_cast(ptr),
                 thrust::device_pointer_cast(ptr + t.getSize()), value);
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void createContiguousCopy(Tensor& res, const Tensor& src) {
    assert(res.getSize() == src.getSize());

    const auto size = src.getSize();
    constexpr int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    const auto ndims = src.getDims().nDims();
    tensorSize_t* d_strides = mempool::tensorSizePool.request(Device::CUDA, ndims);
    cudaErrchk(cudaMemcpy(d_strides, src.getDims().getStrides().data(), ndims* sizeof(tensorSize_t), cudaMemcpyHostToDevice));

    tensorSize_t* d_dims = mempool::tensorDimPool.request(Device::CUDA, ndims);
    cudaErrchk(cudaMemcpy(d_dims, src.getDims().data(), ndims* sizeof(tensorDim_t), cudaMemcpyHostToDevice));

    createContiguousCopyKernel<<<blocks, threadsPerBlock>>>(
      res.getData(), src.getData(), d_strides, d_dims, ndims, size);
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif

    mempool::tensorDimPool.giveback(d_dims, Device::CUDA, ndims);
    mempool::tensorSizePool.giveback(d_strides, Device::CUDA, ndims);
  }

  void getSlice(Tensor& res, const Tensor& src, span<const tensorDim_t> idx) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (res.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    tensorDim_t* idx_d = mempool::tensorDimPool.request(Device::CUDA, idx.size());
    cudaErrchk(cudaMemcpy(idx_d, idx.data(), idx.size() * sizeof(tensorDim_t), cudaMemcpyHostToDevice));

    const auto sizeOfDim = res.getDims().getStride(0);
    getSliceKernel<<<blocks, threadsPerBlock>>>(res.getData(), src.getData(), idx_d, sizeOfDim, res.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif

    mempool::tensorDimPool.giveback(idx_d, Device::CUDA, idx.size());
  }
}
