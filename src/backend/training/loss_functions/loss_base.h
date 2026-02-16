/**
 * @file loss_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"

class LossBase {
  public:
    virtual Tensor operator()(Tensor& y, const Tensor& y_target) const noexcept = 0;
};