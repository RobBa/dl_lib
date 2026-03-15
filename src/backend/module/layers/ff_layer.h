/**
 * @file ff_layer.h
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
#include "utility/initializers.h"

#include <optional>

namespace module {
  class FfLayer : public ModuleBase {
    bool requiresGrad = false;
    bool useBias = false;

    std::shared_ptr<Tensor> weights = nullptr;
    std::shared_ptr<Tensor> bias = nullptr;

  public:
    FfLayer(tensorDim_t inSize, tensorDim_t outSize, bool useBias=true, bool requiresGrad=false);
    FfLayer(tensorDim_t inSize, tensorDim_t outSize, Device d, bool useBias=true, bool requiresGrad=false);

    Tensor operator()(const Tensor& input) const override;
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) const override;

    const Dimension& getDims() const {
      assert(weights);
      return weights->getDims();
    }

    auto getWeights() const noexcept { return weights; }
    auto getBias() const noexcept { return bias; }

    bool hasWeights() const {
      return weights != nullptr;
    }

    std::vector< std::shared_ptr<Tensor> > parameters() const override {
      return {weights, bias};
    }

    void print(std::ostream& os) const noexcept override;
  };
}
