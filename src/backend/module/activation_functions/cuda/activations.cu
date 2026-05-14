/**
 * @file activations.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-31
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "activations.cuh"
#include "utility/cuda/cuda_common.cuh"

#include <stdexcept>

using namespace std;

namespace {
  __global__ void reluKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    constexpr ftype zero = 0;
    res[gid] = fmaxf(input[gid], zero);
  }

  __global__ void leakyReluKernel(ftype* const res, const ftype* const input, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = fmaxf(input[gid], eps * input[gid]); // eps < 1
  }

  __device__ __forceinline__ ftype sigmoid(ftype x) {
      ftype z = expf(-fabsf(x));
      ftype s = 1.0f / (1.0f + z);
      return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
  }

  __global__ void sigmoidKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = sigmoid(input[gid]);
  }

  /**
   * @brief Reduction kernel that computes the maximum within the size of 2 * warpsize at maximum.
   */
  template<int maxoffset>
  __forceinline__ __device__ void warpMaxReduce(volatile ftype* const input, const tensorSize_t stride, const int offset) {
    if(maxoffset == 32) {
      if(offset + 32 < stride) input[offset] = max(input[offset], input[offset + 32]);
    }
    if(maxoffset >= 16) {
      if(offset + 16 < stride) input[offset] = max(input[offset], input[offset + 16]);
    }
    if(maxoffset >= 8) {
      if(offset + 8 < stride) input[offset] = max(input[offset], input[offset + 8]);
    }
    if(maxoffset >= 4) {
      if(offset + 4 < stride) input[offset] = max(input[offset], input[offset + 4]);
    }
    if(maxoffset >= 2) {
      if(offset + 2 < stride) input[offset] = max(input[offset], input[offset + 2]);
    }
    if(maxoffset >= 1) {
      if(offset + 1 < stride) input[offset] = max(input[offset], input[offset + 1]);
    }
  }

  /**
   * @brief Reduction kernel that computes the sum over an array within the size of 2 * warpsize at maximum.
   */
  template<int maxoffset>
  __forceinline__ __device__ void warpSumReduce(volatile ftype* const input, const tensorSize_t stride, const int offset) {
    if(maxoffset == 32) {
      if(offset + 32 < stride) input[offset] += input[offset + 32];
    }
    if(maxoffset >= 16) {
      if(offset + 16 < stride) input[offset] += input[offset + 16];
    }
    if(maxoffset >= 8) {
      if(offset + 8 < stride) input[offset] += input[offset + 8];
    }
    if(maxoffset >= 4) {
      if(offset + 4 < stride) input[offset] += input[offset + 4];
    }
    if(maxoffset >= 2) {
      if(offset + 2 < stride) input[offset] += input[offset + 2];
    }
    if(maxoffset >= 1) {
      if(offset + 1 < stride) input[offset] += input[offset + 1];
    }
  }
  
  /**
   * @brief Here we find the maximum within 'stride'. Assumption: One warp does exactly one element of stride!
   * Reduction via warp reduce. res has the maximum values stored.
   */
  template<int maxoffset>
  __global__ void findMaxKernelOneWarp(ftype* const res, const ftype* const input, const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    int tid = threadIdx.x;
    extern __shared__ ftype smem[];
    smem[tid] = input[gid];
    __syncthreads();

    volatile ftype* const start = smem + (tid / stride) * stride;
    const int offset = gid % stride;
    warpMaxReduce<maxoffset>(start, stride, offset);

    // one warp reduces one 'stride'
    if(offset == 0)
      res[tid / 32] = smem[tid];
  }

  /**
   * @brief Numerically stable version of 
   */
  template<typename T, int maxoffset>
  __global__ void stableSoftmaxKernelOneWarp(ftype* const res, const ftype* const input, const ftype* const maxValues,
                                  const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    const auto maxValue = maxValues[strideOffset];

    int tid = threadIdx.x;
    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>
    __syncthreads();

    ftype expVal = 0;
    if constexpr (std::is_same_v<T, float>) {
      expVal = expf(input[gid] - maxValue);
    }
    else if constexpr (std::is_same_v<T, double>) {
      expVal = exp(input[gid] - maxValue);
    }
    else {
      static_assert(always_false<T>, "ftype encountered unexpected type");
    }
    smem[tid] = expVal;
    __syncthreads();

    volatile ftype* const start = smem + (tid / stride) * stride;
    const int offset = gid % stride;
    warpSumReduce<maxoffset>(start, stride, offset);

    res[gid] = expVal / start[0];
  }

  // TODO: use shared memory
  __global__ void findMaxKernelOneBlock(ftype* res, ftype* const input, const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    int tid = threadIdx.x;
    const tensorSize_t baseOffset = gid % stride;
    for(tensorSize_t offset = (stride + 1) / 2; offset > 32; offset = (offset + 1) / 2) { // ceil-division
      // ceil-division over two -> last thread crosses boundary iff stride % 2 == 1
      if(baseOffset + offset < stride) {
        input[gid] = max(input[gid], input[gid + offset]);
      }
      __syncthreads();
    }

    // loop unrolling
    if(baseOffset + 16 < stride) {
      input[gid] = max(input[gid], input[gid + 16]);
      __syncthreads();
    }
    if(baseOffset + 8 < stride) {
      input[gid] = max(input[gid], input[gid + 8]);
      __syncthreads();
    }
    if(baseOffset + 4 < stride) {
      input[gid] = max(input[gid], input[gid + 4]);
      __syncthreads();
    }    
    if(baseOffset + 2 < stride) {
      input[gid] = max(input[gid], input[gid + 2]);
      __syncthreads();
    }
    if(baseOffset + 1 < stride) {
      input[gid] = max(input[gid], input[gid + 1]);
      __syncthreads();
    }

    // write to result
    if(baseOffset == 0) {
      res[gid / stride] = input[gid];
    }
  }

  // TODO: use shared memory
  template<typename T>
  __global__ void expAndSumKernelOneBlock(T* res, T* const tmp, const T* const input, const T* const maxValues,
                                  const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    const auto maxValue = maxValues[strideOffset];
    if constexpr (std::is_same_v<T, float>) {
      res[gid] = expf(input[gid] - maxValue);
    }
    else if constexpr (std::is_same_v<T, double>) {
      res[gid] = exp(input[gid] - maxValue);
    }
    else {
      static_assert(always_false<T>, "ftype encountered unexpected type");
    }

    __syncthreads();

    const tensorSize_t baseOffset = gid % stride;
    tensorSize_t offset = (stride + 1) / 2; // ceil-division
    while(offset > 32) {
      if(baseOffset + offset >= stride) continue; // ceil-division -> last thread can cross boundary

      tmp[gid] = res[gid] + res[gid + offset];
      offset = (offset + 1) / 2;
    }

    // loop unrolling
    if(offset <= 32 && baseOffset + 16 < stride) {
      tmp[gid] = res[gid] + res[gid + 16];
    }
    if(offset <= 16 && baseOffset + 8 < stride) {
      tmp[gid] = res[gid] + res[gid + 8];
    }
    if(offset <= 8 && baseOffset + 4 < stride) {
      tmp[gid] = res[gid] + res[gid + 4];
    }
    if(offset <= 4 && baseOffset + 2 < stride) {
      tmp[gid] = res[gid] + res[gid + 2];
    }
    if(offset <= 2 && baseOffset + 1 < stride) {
      tmp[gid] = res[gid] + res[gid + 1];
    }

    // write to result
    if(baseOffset == 0) {
      tmp[strideOffset] = tmp[gid];
    }
  }

  __global__ void softmaxDivisionKernel(ftype* res, const ftype* const sums, const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    res[gid] = res[gid] / sums[strideOffset];
  }
}

namespace cuda_impl {
  void relu(Tensor& res, const Tensor& in) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

    reluKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), in.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void leakyRelu(Tensor& res, const Tensor& in, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

    leakyReluKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), eps, in.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void sigmoid(Tensor& res, const Tensor& in) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

    sigmoidKernel<<<blocks, threadsPerBlock>>>(res.getData(), in.getData(), in.getSize());
    cudaErrchk(cudaDeviceSynchronize());
}

  void softmax(Tensor& res, const Tensor& in) {
    const tensorSize_t stride = static_cast<tensorSize_t>(in.getDims().get(-1));
    if(stride == 1) {
      __throw_invalid_argument("Attemped softmax of one element");
    }

    // TODO: use some static struct here to prevent those guys from keeping on re-allocating memory
/*     ftype* tmp;
    cudaErrchk(cudaMalloc(&tmp, in.getSize() * sizeof(ftype)));
    cudaErrchk(cudaMemcpy(tmp, in.getData(), in.getSize() * sizeof(ftype), cudaMemcpyDeviceToDevice)); */

    ftype* maxValues;
    const tensorSize_t nMaxValues = in.getSize() / stride;
    cudaErrchk(cudaMalloc(&maxValues, nMaxValues * sizeof(ftype)));

    static const auto warpSizeT2 = 2 * DeviceProperties::getWarpSize(); // TODO: can this be a problem in a multi-GPU setting?
    if(stride <= warpSizeT2) {
      assert(DeviceProperties::getWarpSize() == 32);

      const int threadsPerBlock = 256;
      const int blocks = (in.getSize() + threadsPerBlock - 1) / threadsPerBlock;

      if(stride == 2) {
        findMaxKernelOneWarp<1> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 4) {
        findMaxKernelOneWarp<2> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 8) {
        findMaxKernelOneWarp<4> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 16) {
        findMaxKernelOneWarp<8> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 32) {
        findMaxKernelOneWarp<16> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      else if(stride <= 64) {
        findMaxKernelOneWarp<32> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      }
      cudaErrchk(cudaDeviceSynchronize());

      if(stride == 2) {
        stableSoftmaxKernelOneWarp<ftype, 1> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 4) {
        stableSoftmaxKernelOneWarp<ftype, 2> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 8) {
        stableSoftmaxKernelOneWarp<ftype, 4> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 16) {
        stableSoftmaxKernelOneWarp<ftype, 8> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 32) {
        stableSoftmaxKernelOneWarp<ftype, 16> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      else if(stride <= 64) {
        stableSoftmaxKernelOneWarp<ftype, 32> <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(ftype)>>>
                                                (res.getData(), in.getData(), maxValues, stride, in.getSize());
      }
      cudaErrchk(cudaDeviceSynchronize());
    }
    else if (stride <= 256) {
      // threads per block needs to be multiple of stride
/*       constexpr int threadsPerBlock = 256;
      const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

      findMaxKernelOneBlock<<<blocks, threadsPerBlock>>>(maxValues, tmp, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      expAndSumKernelOneBlock<ftype><<<blocks, threadsPerBlock>>>(res.getData(), tmp, in.getData(), maxValues, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      softmaxDivisionKernel<<<blocks, threadsPerBlock>>>(res.getData(), tmp, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize()); */
    }
    else {
      __throw_runtime_error("Softmax kernels not yet implemented at inter-block level");
    }

    //cudaErrchk(cudaFree(tmp));
    cudaErrchk(cudaFree(maxValues));
  }
}