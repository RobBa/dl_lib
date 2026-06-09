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

#include "shared/global_params.h"
#include "data_modeling/tensor.h"

namespace cuda_impl {
  void reluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent);
  void leakyReluBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& parent, ftype eps);

  void sigmoidBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& sigmoid);
  void softmaxBackward(Tensor& res, const Tensor& upstreamGrad, const Tensor& softmax);
}
