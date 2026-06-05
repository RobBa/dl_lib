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

#include "utility/global_params.h"

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
  __global__ void powerTwoSumKernel(ftype* const output, const ftype* const grads, const ftype maxNorm, const int idx, const tensorSize_t size) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int gridOffset = gridDim.x * blockDim.x;

    const ftype g1 = gid < size ? grads[gid] : 0.;
    const ftype g2 = gid + gridOffset < size ? grads[gid + gridOffset] : 0.;
    
    extern __shared__ ftype smem[];
    smem[tid] = g1 * g1 + g2 * g2;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 32; offset >>=1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle
    volatile ftype* const sdata = smem;
    if(tid < 32) {
      if(gid + 32 < size) {
        sdata[tid] += sdata[tid + 32];
      }
      if(gid + 16 < size) {
        sdata[tid] += sdata[tid + 16];
      }
      if(gid + 8 < size) {
        sdata[tid] += sdata[tid + 8];
      }
      if(gid + 4 < size) {
        sdata[tid] += sdata[tid + 4];
      }
      if(gid + 2 < size) {
        sdata[tid] += sdata[tid + 2];
      }
      if(gid + 1 < size) {
        sdata[tid] += sdata[tid + 1];
      }
    }

    if(tid == 0) {
      if constexpr (useBlockIdx) {
        output[blockIdx.x] = sdata[0];
      }
      else {
        output[idx] = sdata[0];
      }
    }
  }

  /**
   * @brief Only for gradient clipping!
   * Works like sumReduceKernel, but also normalizes with a sqrt in the end.
   * Writes to output[0], which is also input.
   */
  template<typename T>
  __global__ void sumReduceAndSqrtKernel(ftype* const output, const ftype* const input, const tensorSize_t inputSize) {
    assert_debug(gridDim.x == 1, "This kernel can only be launched in one block");
    assert_debug(blockDim.x <= inputSize, "blockDim.x must be less or equal than size");

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    extern __shared__ ftype smem[]; 
    const ftype x1 = gid < inputSize ? input[gid] : 0;
    const ftype x2 = gid + blockDim.x < inputSize ? input[gid + blockDim.x] : 0;
    smem[tid] = x1 + x2;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
      if(tid < offset) {
        smem[tid] += smem[tid + offset];
      }
      __syncthreads();
    }

    // TODO: warp shuffle
    volatile ftype* const sdata = smem;
    if(tid < 32) {
      if(tid + 32 < inputSize) {
        sdata[tid] += sdata[tid + 32];
      }
      if(tid + 16 < inputSize) {
        sdata[tid] += sdata[tid + 16];
      }
      if(tid + 8 < inputSize) {
        sdata[tid] += sdata[tid + 8];
      }
      if(tid + 4 < inputSize) {
        sdata[tid] += sdata[tid + 4];
      }
      if(tid + 2 < inputSize) {
        sdata[tid] += sdata[tid + 2];
      }
      if(tid + 1 < inputSize) {
        sdata[tid] += sdata[tid + 1];
      }
    }

    if(tid == 0) {
      if constexpr (std::is_same_v<T, float>) {
        output[0] = __fsqrt_rd(sdata[0]);
      } 
      else if constexpr (std::is_same_v<T, double>) {
        output[0] = sqrt(sdata[0]);
      }
      else {
        static_assert(always_false<T>, "Unexpected value for ftype encountered");
      }
    }
  }

  /**
   * @brief Clip the gradients with scale = maxNorm / (totalNorm + EPS_OPTIM_GRADCLIP) when totalNorm > maxNorm.
   */
  __global__ void clipGradientsKernel(ftype* const grads, const ftype* const totalNorm, const ftype maxNorm, const tensorSize_t size) {
    assert(blockDim.x == 1 && gridDim.x == 1);
    
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const ftype tNorm = totalNorm[0];
    const ftype scale = tNorm > maxNorm ? maxNorm / (tNorm + EPS_OPTIM_GRADCLIP) : 1.0;
    
    grads[gid] *= scale;
  }
  
  __global__ void stepSgdKernel(ftype* const params, const ftype* const grads, const ftype lr, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    params[gid] = params[gid] - lr * grads[gid];
  }

  __global__ void stepRmsPropKernel(ftype* const tensor, ftype* const v, const ftype* const grads, const ftype lr, const ftype decay, const tensorSize_t size) {
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
    ftype* totalNorm;
    cudaErrchk(cudaMalloc(&totalNorm, params.size() * sizeof(ftype)));
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
      if(blocks > threadsPerBlock * 2) {
        __throw_invalid_argument("Gradients too large for gradient clipping at the moment.");
      }

      // TODO: parallelize both paths with cuda streams
      if(blocks > 1) {
        ftype* tmp;
        cudaErrchk(cudaMalloc(&tmp, blocks * sizeof(ftype)));

        powerTwoSumKernel<true><<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(
                                  tmp, grads->getData(), maxNorm, /*sentinel value*/ -1, grads->getSize());
        cudaErrchk(cudaDeviceSynchronize());

        const int threadsPerBlock2 = blocks > threadsPerBlock ? threadsPerBlock * 2 : threadsPerBlock;
        sumReduceKernel<<<1, threadsPerBlock2, 2 * threadsPerBlock2 * sizeof(ftype)>>>(
                          totalNorm, tmp, paramIdx, grads->getSize());
        cudaErrchk(cudaDeviceSynchronize());

        cudaErrchk(cudaFree(tmp));
      }
      else {
        powerTwoSumKernel<false><<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(
                                  totalNorm, grads->getData(), maxNorm, paramIdx, grads->getSize());
        cudaErrchk(cudaDeviceSynchronize());
      }
    }

    // step 2: get the total sum of all totalNorm values
    {
      const int threadsPerBlock = params.size() < 512 ? 512 : 1024;
      sumReduceAndSqrtKernel<ftype><<<1, threadsPerBlock, 2 * threadsPerBlock * sizeof(ftype)>>>(
                        totalNorm, totalNorm, params.size());
      cudaErrchk(cudaDeviceSynchronize());
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
      cudaErrchk(cudaDeviceSynchronize());
    }

    cudaErrchk(cudaFree(totalNorm));
  }

  void sgdStep(Tensor& param, const Tensor& grads, ftype lr) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    stepSgdKernel<<<blocks, threadsPerBlock>>>(param.getData(), grads.getData(), lr, param.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void rmspropStep(Tensor& param, Tensor& movingAvg, const Tensor& grads, ftype lr, ftype decay) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (param.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    stepRmsPropKernel<<<blocks, threadsPerBlock>>>(param.getData(), movingAvg.getData(), grads.getData(), lr, decay, param.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }
}
