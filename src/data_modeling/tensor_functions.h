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
#include "initializers.h"

#include <memory>
#include <type_traits>

/**
 * @brief Class providing functions that can be used to create/manipulate tensors.
 * For convenience.
 * 
 * This is defined as a class so we can make some functions private while allowing for
 * templates.
 */
namespace TensorFunctions { // class name acts as namespace for us
    // Tensor creation
    template<typename... T>
    Tensor Zeros(Device d, T... dims) {
      auto res = Tensor(d, dims...);
      res.reset(0);
      return res;
    }
    
    template<typename... T>
    Tensor Zeros(T... dims) {
      return Zeros(Tensor::getDefaultDevice(), dims...);
    }

    template<typename... T>
    Tensor Ones(Device d, T... dims) {
      auto res = Tensor(d, dims...);
      res.reset(1);
      return res;
    }

    template<typename... T>
    Tensor Ones(T... dims) {
      return Ones(Tensor::getDefaultDevice(), dims...);
    }

    template<typename... T>
    Tensor Gaussian(Device d, T... dims) {
      auto res = Tensor(d, dims...);
      res.reset(utility::InitClass::Gaussian);
      return res;
    }
    
    template<typename... T>
    Tensor Gaussian(T... dims) {
      return Gaussian(Tensor::getDefaultDevice(), dims...);
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