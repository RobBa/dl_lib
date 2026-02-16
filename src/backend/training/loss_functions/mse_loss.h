/**
 * @file mse_loss.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "loss_base.h"

class MseLoss final : private LossBase {
  private:
    Tensor operator()(Tensor& y, const Tensor& y_target) const noexcept override;

  public:
    MseLoss() = default;
};
