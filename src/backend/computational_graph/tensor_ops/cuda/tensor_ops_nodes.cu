/**
 * @file tensor_ops_nodes.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include "tensor_ops_nodes.cuh"
#include "utility/cuda/cuda_common.cuh"

using namespace std;

namespace {
  __global__ void scalarMulKernel(ftype* const res, const ftype* const upstreamGrad, const ftype factor, const tensorSize_t size) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;
    res[gid] = upstreamGrad[gid] * factor;
  }
}

namespace cuda_impl {
  void scalarMulBackward(Tensor& res, const Tensor& upstreamGrad, ftype factor) {
    constexpr int threadsPerBlock = 256;
    const int blocks = (upstreamGrad.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    scalarMulKernel<<<blocks, threadsPerBlock>>>(res.getData(), upstreamGrad.getData(), factor, upstreamGrad.getSize());
    cudaErrchk(cudaDeviceSynchronize());
  }

  void getterBackward(Tensor& res, ftype val, tensorSize_t linearIdx) {
    cudaErrchk(cudaMemset(res.getData(), 0, res.getSize() * sizeof(ftype)));
    cudaErrchk(cudaDeviceSynchronize());

    cudaErrchk(cudaMemcpy(res.getData() + linearIdx, &val, sizeof(ftype), cudaMemcpyHostToDevice));
    cudaErrchk(cudaDeviceSynchronize());
  }
}
