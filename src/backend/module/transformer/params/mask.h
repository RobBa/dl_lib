/**
 * @file mask.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

namespace module {
  namespace transformer {
    /**
     * @brief Represents the kind of mask to use on the attention.
     * 
     */
    enum class MaskEnum {
      None,
      Causal,
    };
  }
}
