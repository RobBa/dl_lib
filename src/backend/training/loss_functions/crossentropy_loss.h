/**
 * @file crossentropy_loss.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "loss_base.h"

namespace train {
  class CrossEntropyLoss final : public LossBase {
    public:
      Tensor operator()(const Tensor& y, const Tensor& ypred) const override;
  };
}
