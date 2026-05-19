/**
 * @file activation_nodes.cu
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

#include "activation_nodes.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  /**
   * @brief Relu backward kernel.
   */
  __global__ void reluBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const parent, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;

    res[gid] =  parent[gid] > 0 ? upstreamGrad[gid] : 0;
  }

  /**
   * @brief Leaky relu backward kernel.
   */
  __global__ void leakyReluBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const parent, const ftype eps, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;

    res[gid] = parent[gid] > 0 ? upstreamGrad[gid] : eps * upstreamGrad[gid];
  }

  /**
   * @brief Sigmoid backward kernel, optimized by using the forward sigmoid.
   */
  __global__ void sigmoidBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const sigmoid, const tensorSize_t size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;

    ftype si = sigmoid[gid];
    res[gid] = si * (1 - si) * upstreamGrad[gid];
  }

  __global__ void softmaxBackwardKernelOneWarp(ftype* const res, const ftype* const softmax, const ftype* const upstreamGrad, 
                                                const tensorSize_t stride, tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
      return;

    const int tid = threadIdx.x;
  }

  /**
   * @brief Softmax backward kernel. This kernel is different than others since it is warp aligned. The inner loop avoids shared memory bank 
   * conflicts by broadcasting.
   * 
   * stridesWidthPerBlock is an awkward name. It is the product of number of strides per block (times) stride. We pre-compute it on host. 
   */
  __global__ void softmaxBackwardKernelOneBlock(ftype* const res, const ftype* const softmax, const ftype* const upstreamGrad, 
                                                const tensorSize_t stride, const int stridesWidthPerBlock, const int threadsPerStride, tensorSize_t size) {
    const int tid = threadIdx.x;

    const int strideNumber = (tid / stride);
    if(tid - (strideNumber * threadsPerStride) > stride) {
      // this warp is padded, i.e. it only exists to align warps with strides
      return;
    }

    const int gid = blockIdx.x * stridesWidthPerBlock + tid;
    const ftype yi = softmax[gid];

    const int strideOffset = strideNumber * stride;
    const int withinStrideOffset = tid % threadsPerStride;
    const int smemOffset = strideOffset + withinStrideOffset;

    extern __shared__ ftype smem[];
    smem[smemOffset] = yi;
    smem[smemOffset + stride] = upstreamGrad[gid];
    __syncthreads();


    ftype grad = 0;
    for(int j = 0; j < stride; j++) {
      // warp alignment -> smem-reads are broadcasted per warp -> no bank conflicts
      ftype yj = smem[strideOffset + j];
      ftype gj = smem[strideOffset + j + stride];

      auto jacobian = (withinStrideOffset == j) ? yi * (1 - yj) : -yi * yj;
      grad += gj * jacobian;
    }

    res[gid] = grad;
  }

  __global__ void softmaxBackwardKernelLargePass1(ftype* const res, const ftype* const softmax, const ftype* const upstreamGrad, 
                                                  const tensorSize_t stride, tensorSize_t size) {
    // TODO: code here
  }
}

namespace cuda_impl {
  void reluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    reluBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), parent.getData(), res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void leakyReluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), parent.getData(), eps, res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void sigmoidBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& sigmoid) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    sigmoidBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), sigmoid.getData(), res.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  /**
   * @brief The backward of the softmax. Due to optimization this function distinguishes three cases of stride size, where stride
   * is the size of the dimension the softmax operation is applied to. The two cases are a stride either fitting into one block or not.
   */
  void softmaxBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& softmax) {
    assert(upstreamGrad.getSize() == softmax.getSize());

    constexpr int maxThreadsPerBlock = 256;
    const int stride = softmax.getDims()[-1];
    
    if(stride < maxThreadsPerBlock) {
      const int threadsPerStride = max(1, ((stride + 31) / 32)) * 32; // == warps per stride * 32
      const int stridesPerBlock = maxThreadsPerBlock / threadsPerStride;
      const int strideWidthPerBlock = stridesPerBlock * stride; // for smem idx computation

      int threadsPerBlock = 1;
      while(threadsPerBlock < threadsPerStride * stridesPerBlock) threadsPerBlock <<= 1; 
      // threadsPerBlock now larger than threadsPerStride * stridesPerBlock

      const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;
      softmaxBackwardKernelOneBlock<<<threadsPerBlock, blocks, 2 * strideWidthPerBlock * sizeof(ftype)>>>(
          res.getData(), upstreamGrad.getData(), softmax.getData(), stride, strideWidthPerBlock, threadsPerStride, softmax.getSize());
      cudaErrchk(cudaDeviceSynchronize());
    }
    else {
      __throw_runtime_error("Not implemented yet");
      // TODO: do multi pass kernel
    }    
  }
}
