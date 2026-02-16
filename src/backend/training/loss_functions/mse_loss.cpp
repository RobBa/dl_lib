/**
 * @file mse_loss.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "mse_loss.h"
#include "utility/global_params.h"

#include <cmath>

/**
 * @brief Expects shape (b-size, 1), or simply (batch-size)
 * 
 * @param y Predicted output
 * @param t_target Target
 * @return Tensor of shape (b-size, 1)
 */
Tensor MseLoss::operator()(Tensor& y, const Tensor& y_target) const noexcept {
  auto res = Tensor(y);
  for(tensorSize_t i = 0; i<y.getSize(); i++){
    res[i] = y.get(i) - y_target.get(i);
  }

  for(tensorSize_t i = 0; i<y.getSize(); i++){
    res[i] = std::sqrt(res[i] * res[i]);
  }

  return res;
}