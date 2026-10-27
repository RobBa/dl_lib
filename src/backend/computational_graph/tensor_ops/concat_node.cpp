/**
 * @file concat_node.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "concat_node.h"

#include <cstring>

#ifdef __CUDA
#include "cuda_runtime.h"
#include "utility/cuda/cuda_common.cuh"
#endif

using namespace std;
using namespace cgraph;

namespace {
  // To better understand the internals of this method I refer to the forward pass implementation.
  void copyValues(Tensor& target, const ftype* const srcValues, 
                  const Device d, const tensorSize_t srcOffset, 
                  const tensorSize_t srcDimSize, const tensorSize_t tgtDimSize) {
    
    ftype* const tgtValues = target.data();
    const auto nCopies = target.getSize() / tgtDimSize;
    
    switch(d) {
      case Device::CPU:
        for(int i = 0; i < nCopies; i++) {
          std::memcpy(tgtValues + i * tgtDimSize, srcValues + i * srcDimSize + srcOffset, tgtDimSize * sizeof(ftype));
        }
        break;
      case Device::CUDA:
      #ifdef __CUDA
        for(int i = 0; i < nCopies; i++) {
          cudaErrchk(cudaMemcpy(tgtValues + i * tgtDimSize, srcValues + i * srcDimSize + srcOffset, 
                                tgtDimSize * sizeof(ftype), cudaMemcpyDeviceToDevice));
        }
      #else 
        throw runtime_error("Received CUDA tensor in non-CUDA compiled version.");
      #endif
        break;
      default:
        throw std::invalid_argument("Encountered unexpected device.");
    }
  }
}

/**
 * To better understand the internals of this method I refer to the forward pass implementation.
 */
std::vector<std::shared_ptr<Tensor>> ConcatNode::backward(const Tensor& upstreamGrad) {
  assert(!upstreamGrad.getRequiresGrad());

  std::vector<std::shared_ptr<Tensor>> res;
  res.reserve(parents.size());

  if(dim > 0) [[likely]] {
    const auto srcDimSize = upstreamGrad.getDims().getStride(dim - 1);

    tensorSize_t upstreamGradOffset = 0; // similar to forward pass
    for(const auto& p : parents) {
      assert(p->getRequiresGrad() && p->getDevice() == upstreamGrad.getDevice()); // device check can circumvented by updating copy-function

      const auto& tgtDims = p->getDims();
      auto grad = make_shared<Tensor>(tgtDims, p->getDevice());
      const auto tgtDimSize = tgtDims.getStride(dim - 1);

      copyValues(*grad, upstreamGrad.data(), upstreamGrad.getDevice(), upstreamGradOffset, srcDimSize, tgtDimSize);
      res.push_back(std::move(grad));

      upstreamGradOffset += tgtDimSize;
    }
  }
  else [[unlikely]] {
    tensorSize_t upstreamGradOffset = 0; // similar to forward pass
    for(const auto& p : parents) {
      assert(p->getRequiresGrad() && p->getDevice() == upstreamGrad.getDevice());

      res.emplace_back(make_shared<Tensor>(upstreamGrad.createView(p->getDims(), upstreamGradOffset)));
      upstreamGradOffset += p->getSize();
    }
  }

  return res;
}