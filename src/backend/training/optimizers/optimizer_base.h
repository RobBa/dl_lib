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

namespace train {
  class OptimizerBase {
    public:
      OptimizerBase() = default;
      ~OptimizerBase() noexcept = default;

      OptimizerBase(const OptimizerBase& other) = delete;
      OptimizerBase& operator=(const OptimizerBase& other) = delete;

      OptimizerBase(OptimizerBase&& other) noexcept = default;
      OptimizerBase& operator=(OptimizerBase&& other) noexcept = default;
      
      virtual void step() = 0;
};
}