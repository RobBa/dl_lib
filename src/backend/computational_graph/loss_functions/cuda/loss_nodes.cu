/**
 * @file loss_nodes.cu
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

#include "loss_nodes.cuh"
#include "module/activation_functions/softmax.h"

#include "shared/cuda/common_kernels.cuh"
#include "utility/cuda/cuda_common.cuh"
#include "utility/global_params.h"

#include <type_traits>

using namespace std;

namespace {
  using namespace cuda_impl;
  
  /**
   * @brief Does what you think it does.
   */
  __global__ void bceBackwardKernel(ftype* const res, const ftype* const yPred, const ftype* const yTrue, const ftype bSize, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const auto yi = yTrue[gid];
    const auto yiHat = yPred[gid];

    const auto g = -yi / cudaMax<ftype>(yiHat, EPS_BCE) + (1 - yi) / cudaMax<ftype>(1 - yiHat, EPS_BCE);
    res[gid] = g / bSize;
  }

  /**
   * @brief BCE backward kernel appplied with sigmoid.
   * 
   * @param bSize Batch-size.
   * @param sigmoids Sigmoids from forward pass.
   */
  __global__ void bceSigmoidBackwardKernel(ftype* const res, const ftype* const logits, const ftype* const yTrue, const ftype bSize, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const auto y = yTrue[gid];
    const auto s = cudaSigmoid(logits[gid]);

    const auto g = s - y;
    res[gid] = g / bSize;
  }

  /**
   * @brief The simple cross-entropy backward kernel.
   * 
   * @param bSize The batch-size.
   */
  __global__ void crossEntropyBackwardKernel(ftype* const res, const ftype* const yPred, const ftype* const yTrue, const ftype nSamples, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    const auto g = -yTrue[gid] / cudaMax<ftype>(yPred[gid], EPS_CROSSENTROPY);
    res[gid] = g / nSamples;
  }

  /**
   * @brief Does what you think it does.
   */
  __global__ void crossEntropySoftmaxBackwardKernel(ftype* const res, const ftype* const softmaxedLogits, const ftype* const yTrue, 
                                                    const ftype nSamples, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }

    res[gid] = (softmaxedLogits[gid] - yTrue[gid]) / nSamples;
  }

  /**
   * @brief RMSE backward kernel. bSize = batch-size, rmse = the rmse from the forward pass.
   */
  __global__ void rmseBackwardKernel(ftype* const res, const ftype* const yPred, const ftype* const yTrue,
                                     const ftype rmse, const ftype bSize, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) {
      return;
    }
    
    const ftype yi = yTrue[gid];
    const ftype yiHat = yPred[gid];

    const ftype denom = rmse * bSize + EPS_RMSE;
    const ftype g = (yiHat-yi) / denom;

    res[gid] = g;
  }
}

namespace cuda_impl {
  void bceBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    bceBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), yPred.getData(), yTrue.getData(), yPred.getDims()[0], yTrue.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void bceSigmoidBackward(Tensor& res, const Tensor& logits, const Tensor& yTrue) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (logits.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    bceSigmoidBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), logits.getData(), yTrue.getData(), logits.getDims()[0], logits.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropyBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    const tensorSize_t stride = yPred.getDims()[-1];
    const ftype nSamples = static_cast<ftype>(yPred.getSize() / stride);

    crossEntropyBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), yPred.getData(), yTrue.getData(), nSamples, yTrue.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void crossEntropySoftmaxBackward(Tensor& res, const Tensor& logits, const Tensor& yTrue) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (logits.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    static const auto softmax = module::Softmax();
    const auto softmaxedLogits = softmax(logits);

    const tensorSize_t stride = logits.getDims().get(-1);
    const auto nSamples = static_cast<ftype>(logits.getSize() / stride);

    crossEntropySoftmaxBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), softmaxedLogits.getData(), yTrue.getData(), nSamples, logits.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void rmseBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue, ftype rmse) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (yPred.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    rmseBackwardKernel<<<blocks, threadsPerBlock>>>(res.getData(), yPred.getData(), yTrue.getData(), rmse, static_cast<ftype>(yPred.getDims()[0]), yPred.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }
}
