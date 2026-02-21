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
#include "utility/initializers.h"

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
  Tensor Zeros(std::vector<tensorDim_t> dims, Device d, const bool requiresGrad=false);
  Tensor Zeros(std::vector<tensorDim_t> dims, const bool requiresGrad=false);

  Tensor Ones(std::vector<tensorDim_t> dims, Device d, const bool requiresGrad=false);
  Tensor Ones(std::vector<tensorDim_t> dims, const bool requiresGrad=false);

  Tensor Gaussian(std::vector<tensorDim_t> dims, Device d, const bool requiresGrad=false);
  Tensor Gaussian(std::vector<tensorDim_t> dims, const bool requiresGrad=false);

  std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims, bool requiresGrad=false);

  std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims, Device d, bool requiresGrad=false);

  std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims, 
                                           const std::vector<ftype>& initValues, 
                                           bool requiresGrad=false);

  std::shared_ptr<Tensor> makeSharedTensor(const std::vector<tensorDim_t>& dims, 
                                           const std::vector<ftype>& initValues, 
                                           Device d, bool requiresGrad=false);

  // Tensor manipulation
  void ToZeros(Tensor& t);
  void ToOnes(Tensor& t);
  void ToGaussian(Tensor& t);
}