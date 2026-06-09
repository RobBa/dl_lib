/**
 * @file loss_nodes.cuh
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
  void bceBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue);
  void bceSigmoidBackward(Tensor& res, const Tensor& logits, const Tensor& yTrue);

  void crossEntropyBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue);
  void crossEntropySoftmaxBackward(Tensor& res, const Tensor& logits, const Tensor& yTrue);

  void rmseBackward(Tensor& res, const Tensor& yPred, const Tensor& yTrue, ftype rmse);
}
