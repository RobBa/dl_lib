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

#include "layers/layer_base.h"
#include "activation_functions/activation_function_base.h"

#include <vector>
#include <memory>

class SequentialNetwork {
  protected:
    std::vector< std::shared_ptr<layers::LayerBase> > layers;

    bool assertDims(const layers::LayerBase& layer) const noexcept;

    void append(std::shared_ptr<layers::LayerBase> l);
    void append(std::shared_ptr<activation::ActivationFunctionBase> f);

  public:
    SequentialNetwork() = default;

    Tensor forward(const Tensor& input) const;
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) const;
};