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

#include <ostream>

class Tensor;

namespace cuda_impl {

  // scalar ops
  void scalaradd(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size);
  void scalarmul(ftype* res, const ftype* const left, ftype scalar, tensorSize_t size);

  // matrix ops
  void elementwiseadd(ftype* res, const ftype* const left, const ftype* const right, tensorSize_t size);
  void broadcastadd(Tensor& res, const Tensor& matrix, const Tensor& vec);
  void elementwisemul(ftype* res, const ftype* const left, const ftype* const right, tensorSize_t size);
  void matmul(Tensor& res, const Tensor& left, const Tensor& right);

  void transpose2D(ftype* res, const ftype* const src, Dimension dims, tensorDim_t dim1, tensorDim_t dim2);
  void transpose(ftype* res, const ftype* const src, Dimension dims, tensorDim_t dim1, tensorDim_t dim2);

  void scalarFill(ftype* ptr, ftype value, tensorSize_t size);

  void printValues(std::ostream& os, const Tensor& t);

  // other
  ftype get(const ftype* const t, tensorSize_t idx);
  ftype set(ftype value, const ftype* t, tensorSize_t idx);

  void createContiguousCopy(Tensor& res, const Tensor& src);
}
