/**
 * @file memory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "global_params.h"

struct MemoryLayout final {
  // aligning memory allows us unsafe/faster memfetches into AVX registers
  constexpr static unsigned int CPU_TENSOR_ALIGNMENT = 64;
  
  // assumption needed e.g. for avoiding false-sharing
  constexpr static unsigned int CACHE_LINE_BYTES = 64; 
};