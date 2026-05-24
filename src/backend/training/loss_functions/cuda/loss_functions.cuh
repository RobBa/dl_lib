/**
 * @file loss_functions.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-10
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "data_modeling/tensor.h"

namespace cuda_impl {
  void bceLoss(Tensor& res, const Tensor& y, const Tensor& yPred);
  void bceSigmoidLoss(Tensor& res, const Tensor& y, const Tensor& logits);

  void crossEntropyLoss(Tensor& res, const Tensor& y, const Tensor& yPred);
  void crossEntropySoftmaxLoss(Tensor& res, const Tensor& y, const Tensor& yPred);

  void rmseLoss(Tensor& res, const Tensor& y, const Tensor& yPred);
}
