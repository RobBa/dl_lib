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

namespace cuda {
  Tensor&& bceLoss(const Tensor& y, const Tensor& yPred);
  Tensor&& bceSigmoidLoss(const Tensor& y, const Tensor& yPred);

  Tensor&& crossEntropyLoss(const Tensor& y, const Tensor& yPred);
  Tensor&& crossEntropySoftmaxLoss(const Tensor& y, const Tensor& yPred);

  Tensor&& rmseLoss(const Tensor& y, const Tensor& yPred);
}