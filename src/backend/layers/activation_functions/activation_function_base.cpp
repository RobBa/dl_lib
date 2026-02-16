/**
 * @file activation_function_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-02
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "activation_function_base.h"

using namespace activation;

Tensor ActivationFunctionBase::forward(Tensor& t) const noexcept {
  return (*this)(t);
}