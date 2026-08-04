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

#include "utility/utils.h"

#if defined(USE_AVX512)
static_assert(false, 
  "This version currently does not support AVX-512 due to hardware not accessible. Recompile with a lower version"
);
#endif // defined(USE_AVX512)

namespace utility {
  struct DLLIB_API AvxInfo final {
    private:
      inline static bool avxAvailable = false;

    public:
      AvxInfo() = delete;
      ~AvxInfo() noexcept = delete;

      static bool getAvxAvailable() noexcept {
        return avxAvailable;
      }

      static void verifyAvxSupport();
  };
}
