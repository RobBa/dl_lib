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
  __global__ void reluKernel(ftype* res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    constexpr ftype zero = 0;
    res[gid] = fmaxf(input[gid], zero);
  }

  __global__ void leakyReluKernel(ftype* res, const ftype* const input, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = fmaxf(input[gid], eps*input[gid]); // eps < 1
  }

  __device__ __forceinline__ ftype sigmoid(ftype x) {
      ftype z = expf(-fabsf(x));
      ftype s = 1.0f / (1.0f + z);
      return (x >= 0.f) ? s : z * s; // x < 0 => e^x/(e^x+1) 
  }

  __global__ void sigmoidKernel(ftype* res, const ftype* const input, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    res[gid] = sigmoid(input[gid]);
  }

  // TODO: use shared memory
  __global__ void findMaxKernel(ftype* res, ftype* const input, const tensorSize_t stride, const tensorSize_t size) {
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
  __global__ void expAndSumKernel(ftype* res, ftype* const tmp, const ftype* const input, const ftype* const maxValues, 
                                  const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    const auto maxValue = maxValues[strideOffset];
    if constexpr (std::is_same_v<ftype, float>) {
      res[gid] = expf(input[gid] - maxValue);
    }
    else if constexpr (std::is_same_v<ftype, double>) {
      res[gid] = exp(input[gid] - maxValue);
    }
    else {
      static_assert(always_false<ftype>, "ftype encountered unexpected type");
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

  __global__ void softmaxDivisionKernel(ftype* res, const ftype* const tmp, const tensorSize_t stride, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const auto strideOffset = gid / stride;
    res[gid] = res[gid] / tmp[strideOffset];
  }
}


namespace cuda {
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

    if (stride <= 1 << 10) {
      // threads per block needs to be multiple of stride
      int threadsPerBlock = 256 / stride * stride; // 256 >= threadsPerBlock >= 256 - stride + 1
      const int blocks = (in.getSize()+threadsPerBlock-1) / threadsPerBlock;

      ftype* tmp;
      cudaErrchk(cudaMalloc(&tmp, in.getSize() * sizeof(ftype)));
      cudaErrchk(cudaMemcpy(tmp, in.getData(), in.getSize() * sizeof(ftype), cudaMemcpyDeviceToDevice));

      ftype* maxValues;
      const tensorSize_t nMaxValues = in.getSize() / stride;
      cudaErrchk(cudaMalloc(&maxValues, nMaxValues * sizeof(ftype)));

      findMaxKernel<<<blocks, threadsPerBlock>>>(maxValues, tmp, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      expAndSumKernel<<<blocks, threadsPerBlock>>>(res.getData(), tmp, in.getData(), maxValues, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      softmaxDivisionKernel(res.getData(), tmp, stride, in.getSize());
      cudaErrchk(cudaDeviceSynchronize());

      cudaErrchk(cudaFree(tmp));
      cudaErrchk(cudaFree(maxValues));
    }
    else {
      __throw_runtime_error("Softmax kernels not yet implemented at inter-block level");
    }
  }
}