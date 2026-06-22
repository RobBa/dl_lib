/**
 * @file test_avx.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <stdexcept>
#include <string_view>

namespace utility {

/* inline std::string_view avxLevel() {
#if defined(USE_AVX512)
  return "AVX512";
#elif defined(USE_AVX2)
  return "AVX2";
#elif defined(USE_AVX)
  return "AVX";
#else
  return "SCALAR";
#endif
} */

inline void checkAvxSupport() {
#if defined(USE_AVX512)
  if (!__builtin_cpu_supports("avx512f")) {
    throw std::runtime_error(
      "Binary compiled with USE_AVX512 but the CPU does not support AVX-512. "
      "Reconfigure with -DAVX_VERSION=AVX2 (or lower) and rebuild.");
  }
#elif defined(USE_AVX2)
  if (!__builtin_cpu_supports("avx2")) {
    throw std::runtime_error(
      "Binary compiled with USE_AVX2 but the CPU does not support AVX2. "
      "Reconfigure with -DAVX_VERSION=AVX (or SCALAR) and rebuild.");
  }
#elif defined(USE_AVX)
  if (!__builtin_cpu_supports("avx")) {
    throw std::runtime_error(
      "Binary compiled with USE_AVX but the CPU does not support AVX. "
      "Reconfigure with -DAVX_VERSION=SCALAR and rebuild.");
  }
#endif
}

}
