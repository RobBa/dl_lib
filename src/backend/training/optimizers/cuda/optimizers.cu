/**
 * @file optimizers.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-13
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "optimizers.cuh"
#include "utility/cuda/cuda_common.cuh"
#include "shared/cuda/common_kernels.cuh"

#include "shared/global_params.h"
#include "shared/memory_pool.h"

#include <stack>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

using namespace std;

namespace {
  using namespace cuda_impl;

  /**
   * @brief Reduction kernel for gradient clipping. Squares input, then does a sum over the input.
   * 
   * There are two versions of this kernel, determined by useBlockIdx template param. If true, then the 
   * final write into output is determined by the blockIdx.x value. We use this variant when a param 
   * (a tensor of gradients) does not fit into one single block. If it does fit into one single block,
   * then useBlockIdx should be false, and we write into output based on input parameter idx.
   * 
   * @param idx The index to write into.
   */
  template<bool useBlockIdx>
  __global__ void powerTwoSumKernel(ftype* __restrict__ const output, const ftype* __restrict__ const grads, const ftype maxNorm, const int idx, const tensorSize_t size) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int gridOffset = gridDim.x * blockDim.x;

    const ftype g1 = gid < size ? grads[gid] : 0.0f;
    const ftype g2 = gid + gridOffset < size ? grads[gid + gridOffset] : 0.0f;
    
    extern __shared__ ftype smem[];
    smem[tid] = g1 * g1 + g2 * g2;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 16; offset >>=1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid < 32) {
      assert(blockDim.x >= 32);

      ftype sum = smem[tid];
      for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if(tid == 0) {
        if constexpr (useBlockIdx) {
          output[blockIdx.x] = sum;
        }
        else {
          output[idx] = sum;
        }
      }
    }
  }

  /**
   * @brief Only for gradient clipping!
   * Works like sumReduceKernel, but also normalizes with a sqrt in the end.
   * Writes to output[0], which is also input.
   */
  template<typename T>
  __global__ void sumReduceAndSqrtKernel(ftype* __restrict__ const output, const ftype* __restrict__ const input, const tensorSize_t inputSize) {
    assert(gridDim.x == 1);

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ ftype smem[]; 
    const ftype x1 = gid < inputSize ? input[gid] : 0.0f;
    const ftype x2 = gid + blockDim.x < inputSize ? input[gid + blockDim.x] : 0.0f;
    smem[tid] = x1 + x2;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 16; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    if(tid < 32) {
      assert(blockDim.x >= 32);

      ftype sum = smem[tid];
      for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if(tid == 0) {
        if constexpr (std::is_same_v<T, float>) {
          output[0] = __fsqrt_rd(sum);
        } 
        else if constexpr (std::is_same_v<T, double>) {
          output[0] = sqrt(sum);
        }
        else {
          static_assert(always_false<T>, "Unexpected value for ftype encountered");
        }
      }
    }
  }

  /**
   * @brief Clip the gradients with scale = maxNorm / (totalNorm + EPS_OPTIM_GRADCLIP) when totalNorm > maxNorm.
   */
  __global__ void clipGradientsKernel(ftype* __restrict__ const grads, const ftype* __restrict__ const totalNorm, const ftype maxNorm, const tensorSize_t size) {    
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const ftype tNorm = totalNorm[0];
    const ftype scale = tNorm > maxNorm ? maxNorm / (tNorm + EPS_OPTIM_GRADCLIP) : 1.0;
    
    grads[gid] *= scale;
  }
  
  __global__ void stepSgdKernel(ftype* __restrict__ const params, const ftype* __restrict__ const grads, const ftype lr, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    params[gid] = params[gid] - lr * grads[gid];
  }

  __global__ void stepRmsPropKernel(ftype* __restrict__ const tensor, ftype* __restrict__ const v, const ftype* __restrict__ const grads, const ftype lr, const ftype decay, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const ftype g = grads[gid];
    const ftype mavg = decay * v[gid] + (1 - decay) * g * g;
    v[gid] = mavg;

    const ftype update = tensor[gid] - (lr * g / (cudaSqrt<ftype>(mavg) + EPS_RMSPROP));
    tensor[gid] = update;
  }
}

namespace cuda_impl {
  /**
   * @brief Clips the gradients of param via a global L2 norm.
   * 
   * @param param In/out parameter.
   */
  void clipGradients(const std::vector< std::shared_ptr<Tensor> >& params, const ftype maxNorm) {
    ftype* totalNorm = mempool::tensorPool.request(Device::CUDA, params.size());
    cudaErrchk(cudaMemset(totalNorm, 0, params.size() * sizeof(ftype)));

    if(params.size() > 1024) {
      __throw_invalid_argument("Too many gradients. Not supported in gradient clipping, yet.");
    }

    // step 1: compute norm per param and store in totalNorm array
    for(int paramIdx = 0; paramIdx < params.size(); paramIdx++) {
      auto grads = params[paramIdx]->getGrads();
      if (!grads)
        continue;

      constexpr int threadsPerBlock = 256;
      int blocks = (grads->getSize() + threadsPerBlock - 1) / threadsPerBlock;
      blocks = max(1, blocks >> 2); // each thread tries to cover two elements
      if(blocks > (threadsPerBlock << 1)) {
        __throw_invalid_argument("Gradients too large for gradient clipping at the moment.");
      }

      // TODO: parallelize both paths with cuda streams
      if(blocks > 1) {
        ftype* tmp = mempool::tensorPool.request(Device::CUDA, blocks);

        powerTwoSumKernel<true><<<blocks, threadsPerBlock, (threadsPerBlock << 1) * sizeof(ftype)>>>(
                                  tmp, grads->getData(), maxNorm, /*sentinel value*/ -1, grads->getSize());
        #ifndef NDEBUG
        cudaErrchk(cudaDeviceSynchronize());
        #endif

        const int threadsPerBlock2 = blocks > threadsPerBlock ? (threadsPerBlock << 1) : threadsPerBlock;
        sumReduceKernel<<<1, threadsPerBlock2, (threadsPerBlock2 << 1) * sizeof(ftype)>>>(
                          totalNorm, tmp, paramIdx, blocks);
        
        #ifndef NDEBUG
        cudaErrchk(cudaDeviceSynchronize());
        #endif

        mempool::tensorPool.giveback(tmp, Device::CUDA, blocks);
      }
      else {
        powerTwoSumKernel<false><<<blocks, threadsPerBlock, (threadsPerBlock << 1) * sizeof(ftype)>>>(
                                  totalNorm, grads->getData(), maxNorm, paramIdx, grads->getSize());
        #ifndef NDEBUG
        cudaErrchk(cudaDeviceSynchronize());
        #endif
      }
    }

    // step 2: get the total sum of all totalNorm values
    {
      const int threadsPerBlock = params.size() < 512 ? 512 : 1024;
      sumReduceAndSqrtKernel<ftype><<<1, threadsPerBlock, (threadsPerBlock << 1) * sizeof(ftype)>>>(
                                      totalNorm, totalNorm, params.size());

      #ifndef NDEBUG
      cudaErrchk(cudaDeviceSynchronize());
      #endif
    }

    // step 3: scale the gradients
    for(const auto& param : params) {
      auto grads = param->getGrads();
      if (!grads)
        continue;

      constexpr int threadsPerBlock = 256;
      int blocks = (grads->getSize() + threadsPerBlock - 1) / threadsPerBlock;
      blocks = max(1, blocks >> 2); // each thread tries to cover two elements

      clipGradientsKernel<<<blocks, threadsPerBlock>>>(grads->getData(), totalNorm, maxNorm, grads->getSize());
    }

    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
    
    mempool::tensorPool.giveback(totalNorm, Device::CUDA, params.size());
  }

  void sgdStep(Tensor& param, const Tensor& grads, ftype lr) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    stepSgdKernel<<<blocks, threadsPerBlock>>>(param.getData(), grads.getData(), lr, param.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }

  void rmspropStep(Tensor& param, Tensor& movingAvg, const Tensor& grads, ftype lr, ftype decay) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    stepRmsPropKernel<<<blocks, threadsPerBlock>>>(param.getData(), movingAvg.getData(), grads.getData(), lr, decay, param.getSize());
    
    #ifndef NDEBUG
    cudaErrchk(cudaDeviceSynchronize());
    #endif
  }
}
