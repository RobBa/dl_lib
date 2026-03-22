/**
 * @file bce_loss.h
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
  class BceSigmoidLoss final : public LossBase {
    public:
      std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor> y, 
                                         const std::shared_ptr<Tensor> logits) const override;
  };
}
