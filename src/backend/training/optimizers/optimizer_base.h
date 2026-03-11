/**
 * @file optimizer_base.h
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
#include "training/loss_functions/loss_base.h"

#include <memory>
#include <utility>

namespace train {
  class OptimizerBase {
    protected:
      ftype lr;
      std::vector< std::shared_ptr<Tensor> > params;

    public:
      OptimizerBase(std::vector< std::shared_ptr<Tensor> > params, ftype lr) 
        : params{std::move(params)}, lr{lr} {};
      
      ~OptimizerBase() noexcept = default;

      OptimizerBase(const OptimizerBase& other) = delete;
      OptimizerBase& operator=(const OptimizerBase& other) = delete;

      OptimizerBase(OptimizerBase&& other) noexcept = default;
      OptimizerBase& operator=(OptimizerBase&& other) noexcept = default;

      virtual void step() = 0;
  };
}