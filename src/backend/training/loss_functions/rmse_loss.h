/**
 * @file rmse_loss.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "loss_base.h"
#include "utility/utils.h"

namespace train {
  class DLLIB_API RmseLoss final : public LossBase {
    public:
      std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor> y, 
                                         const std::shared_ptr<Tensor> ypred) const override;
  };
}
