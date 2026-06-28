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

struct MemoryLayout final {
  // aligning memory allows us unsafe/faster memfetches into AVX registers
  constexpr static unsigned int CPU_TENSOR_ALIGNMENT = 64;
  
  // assumption needed e.g. for avoiding false-sharing
  constexpr static unsigned int CACHE_LINE_BYTES = 64; 

  // L1 cache size in bytes, depends on CPU. E.g. 48 * (1 << 10) == 48 KB
  constexpr static unsigned int L1_CACHE_BYTES = 48 * (1 << 10);

  // L3 cache size in bytes, depends on CPU. E.g. 12 * (1 << 20) == 12 MB
  constexpr static unsigned int L3_CACHE_BYTES = 12 * (1 << 20);
};