/**
 * @file function_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"

#include <memory>
#include <iostream>

namespace activation {
  class ActivationFunctionBase {
    public:
      ActivationFunctionBase() = default;

      ActivationFunctionBase(const ActivationFunctionBase& other) = delete;
      ActivationFunctionBase& operator=(const ActivationFunctionBase& other) = delete;

      ActivationFunctionBase(ActivationFunctionBase&& other) noexcept = default;
      ActivationFunctionBase& operator=(ActivationFunctionBase&& other) noexcept = default;

      ~ActivationFunctionBase() noexcept = default;

      // creates no graph
      virtual Tensor operator()(const Tensor& t) const = 0;
      
      // greates a graph
      virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& t) const = 0;

      virtual void print(std::ostream& os) const noexcept { };
      friend std::ostream& operator<<(std::ostream& os, const ActivationFunctionBase& t) noexcept;
  };
}
