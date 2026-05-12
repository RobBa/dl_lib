/**
 * @file tensorops.cuh
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
#include "data_modeling/dim_type.h"

class Tensor;

namespace cuda_impl {

  // scalar ops
  void scalaradd(Tensor& res, const Tensor& src, ftype scalar);
  void scalarmul(Tensor& res, const Tensor& src, ftype scalar);

  // matrix ops
  void elementwiseadd(Tensor& res, const Tensor& left, const Tensor& right);
  void broadcastadd(Tensor& res, const Tensor& matrix, const Tensor& vec);
  void elementwisemul(Tensor& res, const Tensor& left, const Tensor& right);
  void matmul(Tensor& res, const Tensor& left, const Tensor& right);

  void transpose2D(ftype* res, const ftype* const src, Dimension dims, tensorDim_t dim1, tensorDim_t dim2);
  void transpose(ftype* res, const ftype* const src, Dimension dims, tensorDim_t dim1, tensorDim_t dim2);

  void scalarFill(Tensor& t, ftype value);

  void createContiguousCopy(Tensor& res, const Tensor& src);
}
