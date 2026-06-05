/**
 * @file optimizers.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-05-13
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "utility/global_params.h"
#include "data_modeling/tensor.h"

#include <vector>
#include <memory>

namespace cuda_impl {
  void clipGradients(const std::vector< std::shared_ptr<Tensor> >& params, ftype maxNorm);

  void sgdStep(Tensor& param, const Tensor& grad, ftype lr);
  void rmspropStep(Tensor& param, Tensor& movingAvg, const Tensor& grad, ftype lr, ftype decay);
}
