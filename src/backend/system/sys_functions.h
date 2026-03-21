/**
 * @file sys_functions.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief A collection of functions determining the behavior of the 
 * whole software system. 
 * @version 0.1
 * @date 2026-01-31
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/device.h"

namespace sys {
  void setDevice(Device d) noexcept;
  Device getDevice() noexcept;

  void setRandomSeed(unsigned int s) noexcept;
}