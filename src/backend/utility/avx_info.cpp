/**
 * @file avx_info.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-09-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "utility/avx_info.h"

#include <iostream>

/**
 * @brief Does what you think it does.
 * 
 * Function cannot be inlined in header, since header does not see the compile time flags.
 */
bool utility::AvxInfo::verifyAvxSupport() {
  #if defined(USE_AVX512)
    if (!__builtin_cpu_supports("avx512f")) [[unlikely]] {
      std::cerr <<
        "Binary compiled with USE_AVX512 but the CPU does not support AVX-512. " <<
        "To use AVX reconfigure with -DAVX_VERSION=AVX2 (or AVX or SCALAR) and rebuild." << std::endl;
      return false;
    }
    return true;
  #elif defined(USE_AVX2)
    if (!__builtin_cpu_supports("avx2")) [[unlikely]] {
      std::cerr <<
        "Binary compiled with USE_AVX2 but the CPU does not support AVX2. " <<
        "To use AVX reconfigure with -DAVX_VERSION=AVX (or SCALAR) and rebuild." << std::endl;
      return false;
    }
    return true;
  #elif defined(USE_AVX)
    if (!__builtin_cpu_supports("avx")) [[unlikely]] {
      std::cerr <<
        "Binary compiled with USE_AVX2 but the CPU does not support AVX. " <<
        "To avoid overhead and suppress this warning reconfigure with -DAVX_VERSION=SCALAR and rebuild." << std::endl;
      return false;
    }
    return true;
  #else
    return false;
  #endif
}
