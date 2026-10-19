/**
 * @file params.h
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
    enum class MaskType {
      None,
      Causal,
    };
  }

  /**
   * @brief We want to instantiate the attention layers in the backend. Therefore we 
   * expose the type of desired attention in the outward constructor layer only.
   * 
   */
  enum class SelfAttentionType {
    ScaledDotProduct,
  };
}
