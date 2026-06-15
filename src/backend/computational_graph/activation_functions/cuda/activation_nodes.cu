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
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    res[gid] =  parent[gid] > 0 ? upstreamGrad[gid] : 0;
  }

  /**
   * @brief Leaky relu backward kernel.
   */
  __global__ void leakyReluBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const parent, const ftype eps, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    res[gid] = parent[gid] > 0 ? upstreamGrad[gid] : eps * upstreamGrad[gid];
  }

  /**
   * @brief Sigmoid backward kernel, optimized by using the forward sigmoid.
   */
  __global__ void sigmoidBackwardKernel(ftype* const res, const ftype* const upstreamGrad, const ftype* const sigmoid, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    ftype si = sigmoid[gid];
    res[gid] = si * (1 - si) * upstreamGrad[gid];
  }

  /**
   * @brief Softmax backward kernel. This kernel is different than others since it is warp aligned. The inner loop avoids shared memory bank 
   * conflicts by broadcasting.
   * 
   * stridesWidthPerBlock is an awkward name. It is the product of number of strides per block (times) stride. We pre-compute it on host. 
   */
  __global__ void softmaxBackwardKernelOneBlock(ftype* const res, const ftype* const upstreamGrad, const ftype* const softmax,  
                                                const tensorSize_t stride, const int stridesWidthPerBlock, const int threadsPerStride, tensorSize_t size) {
    const int tid = threadIdx.x;

    const int withinStrideOffset = tid % threadsPerStride;
    const int strideOffset = (tid / threadsPerStride) * stride;

    const int gid = blockIdx.x * stridesWidthPerBlock + strideOffset + withinStrideOffset;
    const bool isPadded = (withinStrideOffset >= stride) || (gid >= size); // padded threads only exists to align warps with strides

    ftype yi = 0;
    const int smemOffset = strideOffset + withinStrideOffset;

    extern __shared__ ftype smem[];
    if(!isPadded) {
      yi = softmax[gid];
      smem[smemOffset] = yi;
      smem[smemOffset + stridesWidthPerBlock] = upstreamGrad[gid];
    }
    __syncthreads();

    if(isPadded) {
      return;
    }

    ftype grad = 0;
    for(int j = 0; j < stride; j++) {
      // warp alignment -> smem-reads are broadcasted per warp -> no bank conflicts
      ftype yj = smem[strideOffset + j];
      ftype gj = smem[strideOffset + j + stridesWidthPerBlock];

      auto jacobian = (withinStrideOffset == j) ? yi * (1 - yj) : -yi * yj;
      grad += gj * jacobian;
    }

    res[gid] = grad;
  }

  /**
   * @brief Large softmax pass. Because the stride now does not fit into one block anymore we do a grid-stride loop.
   */
  __global__ void softmaxBackwardKernelLargePass(ftype* const res, const ftype* const upstreamGrad, const ftype* const softmax, const int blocksPerStride, const tensorSize_t stride) {    
    const int strideNumber = blockIdx.x / blocksPerStride;
    const int strideOffset = strideNumber * stride;
    const int i = (blockIdx.x % blocksPerStride) * blockDim.x + threadIdx.x;
    // blockIdx.x % blocksPerStride = block number within this stride

    const int tid = threadIdx.x;
    const int gid = strideOffset + i;

    extern __shared__ ftype smem[];

    const bool isNotPadded = i < stride;
    const ftype yi = isNotPadded ? softmax[gid] : 0;

    ftype grad = 0;
    for(int offset = 0; offset < stride; offset += blockDim.x) {
      // load into smem
      {
        const int j = offset + tid;
        if(j < stride) {
          smem[tid] = softmax[strideOffset + j];
          smem[tid + blockDim.x] = upstreamGrad[strideOffset + j];
        }
        __syncthreads();
      }


      for(int k = 0; k < blockDim.x; k++) {
        const int j = offset + k;
        if(j < stride) {
          ftype yj = smem[k];
          ftype gj = smem[k + blockDim.x];

          auto jacobian = (i == j) ? yi * (1 - yj) : -yi * yj;
          grad += gj * jacobian;
        }
      }
      __syncthreads();
    }

    if(isNotPadded) {
      res[gid] = grad;
    } 
  }
}

namespace cuda_impl {
  void reluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    reluBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), parent.getData(), res.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void leakyReluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent, ftype eps) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), parent.getData(), eps, res.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void sigmoidBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& sigmoid) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    sigmoidBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), sigmoid.getData(), res.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
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

      // min over maximum possible strides per block and actual number of strides
      const int stridesPerBlock = min(maxThreadsPerBlock / threadsPerStride, softmax.getSize() / stride);
      const int strideWidthPerBlock = stridesPerBlock * stride; // for smem idx computation
      
      int threadsPerBlock = 1;
      while(threadsPerBlock < threadsPerStride * stridesPerBlock) threadsPerBlock <<= 1; 
      // threadsPerBlock now larger than threadsPerStride * stridesPerBlock
      const int nStrides = softmax.getSize() / stride;
      const int blocks = (nStrides + stridesPerBlock - 1) / stridesPerBlock;

      softmaxBackwardKernelOneBlock<<<blocks, threadsPerBlock, 2 * strideWidthPerBlock * sizeof(ftype)>>>(
          res.getData(), upstreamGrad.getData(), softmax.getData(), stride, strideWidthPerBlock, threadsPerStride, softmax.getSize());
    }
    else {
      constexpr int maxThreadsPerBlock = 256; 

      const int nStrides = softmax.getSize() / stride;
      const int threadsPerBlock = maxThreadsPerBlock; // TODO: do that one better, this can result in gross imbalance; also for normal softmax
      const int blocksPerStride = (stride + threadsPerBlock - 1) / threadsPerBlock; 

      softmaxBackwardKernelLargePass<<<blocksPerStride * nStrides, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(
                                       res.getData(), upstreamGrad.getData(), softmax.getData(), blocksPerStride, stride);
    }

    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }
}
