/**
 * @file activation_nodes.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-23
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "utility/global_params.h"

namespace cuda {
  __global__ void reluBackward(ftype* res, const ftype* const upstreamGrad, tensorSize_t size);
  __global__ void leakyReluBackward(ftype* res, const ftype* const upstreamGrad, ftype eps, tensorSize_t size);

  __global__ void sigmoidBackward(ftype* res, const ftype* const sigmoids, const ftype* const upstreamGrad, tensorSize_t size);
  __global__ void softmaxBackward(ftype* res, const ftype* const softmax, const ftype* const upstreamGrad, tensorSize_t size);
}