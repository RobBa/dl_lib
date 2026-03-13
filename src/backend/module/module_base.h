/**
 * @file module_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"
#include "utility/global_params.h"

#include <optional>
#include <memory>
#include <utility>

#include <iostream>

namespace module {
  /** 
   * The base class for all the layers that we have. Not instantiable.
   */
  class ModuleBase {
    public:
      ModuleBase() = default;

      ModuleBase(const ModuleBase& other) = delete;
      ModuleBase& operator=(const ModuleBase& other) = delete;

      ModuleBase(ModuleBase&& other) noexcept = default;
      ModuleBase& operator=(ModuleBase&& other) noexcept = default;

      ~ModuleBase() noexcept = default;

      // for inference -> no graph creation
      virtual Tensor operator()(const Tensor& input) const = 0;
      // for training -> creates graph
      virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) const = 0;

      virtual std::vector< std::shared_ptr<Tensor> > parameters() const { return {}; }

      virtual void print(std::ostream& os) const noexcept {};
      friend std::ostream& operator<<(std::ostream& os, const ModuleBase& t) noexcept;
  };
}