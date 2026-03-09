/**
 * @file relu.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "activation_function_base.h"

namespace activation {
  class ReLu final : public ActivationFunctionBase {
    public:
      ReLu() = default;
      
      Tensor operator()(const Tensor& t) const override;
      std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& t) const override;
  };
}
