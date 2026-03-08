/**
 * @file sys_functions.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-31
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "sys_functions.h"
#include "data_modeling/tensor.h"

using namespace sys;

void setDevice(Device d) noexcept {
  Tensor::setDefaultDevice(d);
}

Device getDevice() noexcept {
  return Tensor::getDefaultDevice();
}