/**
 * @file sequential.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "module/module_base.h"

#include <vector>
#include <memory>

namespace module {
  class Sequential : public ModuleBase {
    protected:
      std::vector< std::shared_ptr<module::ModuleBase> > layers;

    public:
      Sequential() = default;
          
      Sequential(const Sequential& other) = delete;
      Sequential& operator=(const Sequential& other) = delete;

      Sequential(Sequential&& other) noexcept = default;
      Sequential& operator=(Sequential&& other) noexcept = default;

      ~Sequential() noexcept = default;

      Tensor operator()(const Tensor& input) const override;
      std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) const override;

      std::vector<std::shared_ptr<Tensor>> parameters() const override;

      void append(std::shared_ptr<module::ModuleBase> l);

      void print(std::ostream& os) const noexcept override;
  };
}
