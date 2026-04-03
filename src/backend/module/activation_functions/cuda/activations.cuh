/**
 * @file activations.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-31
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

namespace cuda {
  void relu(Tensor& res, const Tensor& in);
  void leakyRelu(Tensor& res, const Tensor& in, ftype eps);

  void sigmoid(Tensor& res, const Tensor& in);
  void softmax(Tensor& res, const Tensor& in);
}
