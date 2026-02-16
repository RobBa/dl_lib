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

#include "tensor.h"

#include <optional>

namespace activation {
  class ActivationFunctionBase {
    public:
      virtual Tensor operator()(Tensor& t) const noexcept = 0;
      Tensor forward(Tensor& t) const noexcept;

      virtual Tensor gradient(const Tensor& t) noexcept = 0;
  };
}
