/**
 * @file softmax.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "activation_function_base.h"

namespace activation {
  class Softmax final : public ActivationFunctionBase {
    public:
      Tensor operator()(const Tensor& t) const noexcept override;
  };
}
