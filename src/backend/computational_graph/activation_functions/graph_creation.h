/**
 * @file graph_creation.h
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

#include "activation_functions/relu.h"
#include "activation_functions/leaky_relu.h"

#include <memory>

namespace graph {
  std::shared_ptr<Tensor> doActivation(const activation::ReLu& r, const std::shared_ptr<Tensor>& t);
  std::shared_ptr<Tensor> doActivation(const activation::LeakyReLu& r, const std::shared_ptr<Tensor>& t);
}
 