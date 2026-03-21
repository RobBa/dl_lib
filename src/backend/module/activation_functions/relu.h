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

#include "module/module_base.h"

namespace module {
  class ReLu final : public ModuleBase {
    public:
      ReLu() = default;
      
      Tensor operator()(const Tensor& t) const override;
      std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& t) const override;
  };
}
