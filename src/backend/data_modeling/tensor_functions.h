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
    Tensor Zeros(std::vector<tensorDim_t> dims, Device d, const bool requiresGrad=true) {
      auto res = Tensor(std::move(dims), d, requiresGrad);
      res.reset(0);
      return res;
    }
    
    Tensor Zeros(std::vector<tensorDim_t> dims, const bool requiresGrad=true) {
      return Zeros(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
    }

    Tensor Ones(std::vector<tensorDim_t> dims, Device d, const bool requiresGrad=true) {
      auto res = Tensor(std::move(dims), d, requiresGrad);
      res.reset(1);
      return res;
    }
    
    Tensor Ones(std::vector<tensorDim_t> dims, const bool requiresGrad=true) {
      return Ones(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
    }

    Tensor Gaussian(std::vector<tensorDim_t> dims, Device d, const bool requiresGrad=true) {
      auto res = Tensor(std::move(dims), d, requiresGrad);
      res.reset(utility::InitClass::Gaussian);
      return res;
    }
    
    Tensor Gaussian(std::vector<tensorDim_t> dims, const bool requiresGrad=true) {
      return Gaussian(std::move(dims), Tensor::getDefaultDevice(), requiresGrad);
    }

    // Tensor manipulation
    void ToZeros(Tensor& t) {
      t.reset(0);
    }

    void ToOnes(Tensor& t) {
      t.reset(1);
    }

    void ToGaussian(Tensor& t) {
      t.reset(utility::InitClass::Gaussian);
    }
};