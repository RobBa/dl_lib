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
#include "utility/macros.h"

#include <stdexcept>

using namespace std;

namespace {
  /**
   * @brief Kernel for forward ReLU function.
   */
  __global__ void reluKernel(ftype* const res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    constexpr ftype zero = 0;
    res[gid] = fmaxf(input[gid], zero);
  }

  /**
   * @brief Kernel for forward Leaky-ReLU function.
   */
  __global__ void leakyReluKernel(ftype* const res, const ftype* const input, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = fmaxf(input[gid], eps * input[gid]); // eps < 1
  }

  /**
   * @brief Single sigmoid computation.
   */
  __device__ __forceinline__ ftype sigmoid(ftype x) {
      ftype z = expf(-fabsf(x));
      ftype s = 1.0f / (1.0f + z);
      return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
  }

  /**
   * @brief Kernel for forward Sigmoid function.
   */
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
    // TODO: warp shuffle for newer architectures
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
    // TODO: warp shuffle for newer architectures
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
   * @brief For the softmax implementations.
   */
  template<typename T>
  __forceinline__ __device__ ftype stableExp(const ftype x, const ftype maxValue) {
    if constexpr (std::is_same_v<T, float>) {
      return expf(x - maxValue);
    }
    else if constexpr (std::is_same_v<T, double>) {
      return exp(x - maxValue);
    }
    else {
      static_assert(always_false<T>, "ftype encountered unexpected type");
    }
  } 
  
  /**
   * @brief Here we find the maximum within 'stride'. Assumption: One warp does exactly one element of stride!
   * Reduction via warp reduce. res has the maximum values stored.
   */
  template<int maxoffset>
  __global__ void findMaxKernelOneWarp(ftype* const res, const ftype* const input, const tensorSize_t stride, const tensorSize_t size) {
    assert(blockDim.x % 32 == 0);

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
    if(offset == 0) {
      res[tid / 32] = smem[tid];
    }
  }

  /**
   * @brief Numerically stable version of softmax kernel. Just as in findMaxKernelOneWarp we assume that stride <= 2 * warpsize.
   * Numerical stability comes from computing the maximum values per row, see findMaxKernelOneWarp and argument maxValues.
   */
  template<typename T, int maxoffset>
  __global__ void stableSoftmaxKernelOneWarp(ftype* const res, const ftype* const input, const ftype* const maxValues,
                                             const tensorSize_t stride, const tensorSize_t size) {
    assert(blockDim.x % 32 == 0);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    const auto maxValue = maxValues[strideOffset];
    ftype expVal = stableExp<ftype>(input[gid], maxValue);

    int tid = threadIdx.x;
    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>
    smem[tid] = expVal;
    __syncthreads();

    volatile ftype* const start = smem + (tid / stride) * stride;
    const int offset = gid % stride;
    warpSumReduce<maxoffset>(start, stride, offset);

    res[gid] = expVal / start[0];
  }

  /**
   * @brief Like findMaxKernelOneWarp. The difference now is that the input size can be much larger. stride is 
   * 64 < stride <= threadsPerBlock. res has the maximum values stored.
   * 
   * In this initial version we assume one kernel per stride, to make matters simple to understand.
   */
  __global__ void findMaxKernelOneBlock(ftype* const res, const ftype* const input, const tensorSize_t stride, const tensorSize_t size) {
    assert_debug(blockDim.x / stride == 0, "Kernel built for one stride per block, blockDim.x is < stride"); 

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * stride + tid;

    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>

    const tensorSize_t maxIdx = tid + blockDim.x;
    const bool doPadding = maxIdx >= stride;
    if(doPadding) {
      smem[tid] = input[gid];
    }
    else {
      smem[tid] = max(input[gid], input[gid + blockDim.x]);
    }
    __syncthreads();

    for(tensorSize_t offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] = max(smem[tid], smem[tid + offset]);
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) start[tid] = max(start[tid], start[tid + 32]);
    if(tid < 16) start[tid] = max(start[tid], start[tid + 16]);
    if(tid < 8) start[tid] = max(start[tid], start[tid + 8]);
    if(tid < 4) start[tid] = max(start[tid], start[tid + 4]);
    if(tid < 2) start[tid] = max(start[tid], start[tid + 2]);
    if(tid < 1) start[tid] = max(start[tid], start[tid + 1]);

    if(tid == 0) { // one block per stride
      res[blockIdx.x] = start[0];
    }
  }

  /**
   * @brief Just like stableSoftmaxKernelOneWarp, but this one works across a whole block, not just a warp.
   * 
   * In this initial version we assume one kernel per stride, to make matters simple to understand.
   */
  template<typename T>
  __global__ void stableSoftmaxKernelOneBlock(ftype* res, const ftype* const input, const ftype* const maxValues,
                                              const tensorSize_t stride, const tensorSize_t size) {
    assert_debug(blockDim.x / stride == 0, "Kernel built for one stride per block, blockDim.x is < stride"); 

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * stride + tid;

    const auto maxValue = maxValues[blockIdx.x]; // TODO: i can share this one within a warp?

    extern __shared__ ftype smem[]; // can lead to bank conflicts iff std::is_same_v<T, double>

    ftype expVal = stableExp<ftype>(input[gid], maxValue);
    smem[tid] = expVal;
    __syncthreads();

    const tensorSize_t maxIdx = tid + blockDim.x;
    const bool doPadding = maxIdx >= stride;

    ftype expValOffset = 0;
    if(!doPadding) { // some threads will be idle here
      expValOffset = stableExp<ftype>(input[gid + blockDim.x], maxValue);
    }

    smem[maxIdx] = expValOffset;
    __syncthreads();

    for(tensorSize_t offset = blockDim.x; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] = smem[tid] + smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle for newer architectures
    volatile ftype* const start = smem;
    if(tid < 32) start[tid] = start[tid] += start[tid + 32];
    if(tid < 16) start[tid] = start[tid] += start[tid + 16];
    if(tid < 8) start[tid] = start[tid] += start[tid + 8];
    if(tid < 4) start[tid] = start[tid] += start[tid + 4];
    if(tid < 2) start[tid] = start[tid] += start[tid + 2];
    if(tid < 1) start[tid] = start[tid] += start[tid + 1];

    res[gid] = expVal / start[0];
    if(!doPadding) {
      res[gid + blockDim.x] = expValOffset / start[0];
    }
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

    // TODO: use some static struct here to prevent this guy from keeping on re-allocating memory
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
      constexpr int maxThreadsPerBlock = 256;
      assert_debug(stride <= maxThreadsPerBlock, "If you adapt maxThreadsPerBlock you also have to adapt the limit in the if-clause above");

      const int nStrides = in.getSize() / stride;
      constexpr int stridesPerBlock = 1; // if multiple strides per block allowed (adapt kernels!): max(maxThreadsPerBlock / stride, 1);

      // threads per block needs to be multiple of stride, rounded toward the next multiple of 32
      const int paddedStride = ((stride + 31) / 32) * 32;
      const int threadsPerBlock = paddedStride * stridesPerBlock / 2; // over 2 for efficiency in reduction 
      const int blocks = (nStrides + stridesPerBlock - 1) / stridesPerBlock; // gerneralized version iff multiple strides per block allowed

      findMaxKernelOneBlock<<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(maxValues, in.getData(), stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      stableSoftmaxKernelOneBlock<ftype><<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(res.getData(), in.getData(), maxValues, 
                                                                                                       stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      __throw_runtime_error("Softmax kernels not yet implemented at inter-block level");
    }

    cudaErrchk(cudaFree(maxValues));
  }
}