/**
 * @file loss_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"

#include <memory>

namespace train {
  class LossBase {
    public:
      LossBase() = default;

      LossBase(const LossBase& other) = delete;
      LossBase& operator=(const LossBase& other) = delete;

      LossBase(LossBase&& other) noexcept = default;
      LossBase& operator=(LossBase&& other) noexcept = default;

      ~LossBase() noexcept = default;

      virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& y, const std::shared_ptr<Tensor>& ypred) const = 0;
  };
}
