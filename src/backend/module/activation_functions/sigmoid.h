/**
 * @file sigmoid.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "module/module_base.h"

namespace module {
  class Sigmoid final : public ModuleBase {
    public:
      Tensor operator()(const Tensor& t) const override;
      std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& t) const override;
  };
}
