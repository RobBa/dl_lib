/**
 * @file avx_info.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-28
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <iostream>

#if defined(USE_AVX512)
static_assert(false, 
  "This version currently does not support AVX-512 due to hardware not accessible. Recompile with a lower version"
);
#endif // defined(USE_AVX512)

namespace utility {
  struct AvxInfo final {
    private:
      inline static bool avxAvailable = false;

    public:
      AvxInfo() = delete;
      ~AvxInfo() noexcept = delete;

      static bool getAvxAvailable() noexcept {
        return avxAvailable;
      }

      static void verifyAvxSupport() {
      #if defined(USE_AVX512)
        if (!__builtin_cpu_supports("avx512f")) [[unlikely]] {
          std::cerr <<
            "Binary compiled with USE_AVX512 but the CPU does not support AVX-512. " << 
            "To use AVX reconfigure with -DAVX_VERSION=AVX2 (or AVX or SCALAR) and rebuild." << std::endl;
        }
        else [[likely]] {
          avxAvailable = true;
        }
      #elif defined(USE_AVX2)
        if (!__builtin_cpu_supports("avx2")) [[unlikely]] {
          std::cerr <<
            "Binary compiled with USE_AVX2 but the CPU does not support AVX2. " << 
            "To use AVX reconfigure with -DAVX_VERSION=AVX (or SCALAR) and rebuild." << std::endl;
        }
        else [[likely]] {
          avxAvailable = true;
        }
      #elif defined(USE_AVX)
        if (!__builtin_cpu_supports("avx")) [[unlikely]] {
          std::cerr <<
            "Binary compiled with USE_AVX2 but the CPU does not support AVX. " << 
            "To avoid overhead and suppress this warning reconfigure with -DAVX_VERSION=SCALAR and rebuild." << std::endl;
        }
        else [[likely]] {
          avxAvailable = true;
        }
      #endif
      }
  };
}
