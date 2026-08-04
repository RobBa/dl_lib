/**
 * @file tensor_functions.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief A file containing multiple functions that are supposed to be utilized 
 * with Tensors.
 * @version 0.1
 * @date 2026-01-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "tensor.h"
#include "shared/initializers.h"
#include "shared/export.h"

#include <utility>

/**
 * @brief Class providing functions that can be used to create/manipulate tensors.
 * For convenience.
 *
 * This is defined as a class so we can make some functions private while allowing for
 * templates.
 */
namespace TensorFunctions { // class name acts as namespace for us
  // Tensor creation
  DLLIB_API Tensor Zeros(std::vector<tensorDim_t> dims, Device d, bool requiresGrad=false);
  DLLIB_API Tensor Zeros(std::vector<tensorDim_t> dims, bool requiresGrad=false);

  DLLIB_API Tensor Ones(std::vector<tensorDim_t> dims, Device d, bool requiresGrad=false);
  DLLIB_API Tensor Ones(std::vector<tensorDim_t> dims, bool requiresGrad=false);

  DLLIB_API Tensor Gaussian(std::vector<tensorDim_t> dims, ftype stddev, Device d, bool requiresGrad=false);
  DLLIB_API Tensor Gaussian(std::vector<tensorDim_t> dims, ftype stddev=1, bool requiresGrad=false);

  DLLIB_API std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims, bool requiresGrad=false);

  DLLIB_API std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims, Device d, bool requiresGrad=false);

  DLLIB_API std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims,
                                           const std::vector<ftype>& initValues,
                                           bool requiresGrad=false);

  DLLIB_API std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims,
                                           const std::vector<ftype>& initValues,
                                           Device d, bool requiresGrad=false);

  // Tensor manipulation
  DLLIB_API void ToZeros(Tensor& t) noexcept;
  DLLIB_API void ToOnes(Tensor& t) noexcept;
  DLLIB_API void ToGaussian(Tensor& t, ftype stddev);

  // Arithmetics
  DLLIB_API Tensor SumOverDims(const Tensor& t, tensorDim_t dim=0); // default 0 for batch-size
}