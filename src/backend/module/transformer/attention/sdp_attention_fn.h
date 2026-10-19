/**
 * @file scaled_dot_product_attention.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "self_attention_fn_base.h"

namespace module {
  namespace transformer {
    /** 
     * Scaled-dot production self attention.
     */
    class SdpAttentionFn : public SelfAttentionFnBase {
      public:
        SdpAttentionFn() = default;

        SdpAttentionFn(const SdpAttentionFn& other) = delete;
        SdpAttentionFn& operator=(const SdpAttentionFn& other) = delete;

        SdpAttentionFn(SdpAttentionFn&& other) noexcept = default;
        SdpAttentionFn& operator=(SdpAttentionFn&& other) noexcept = default;

        ~SdpAttentionFn() noexcept = default;

        Tensor operator()(const Tensor& K, const Tensor& Q, const Tensor& V, transformer::MaskType mask) override;
        std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& K, const std::shared_ptr<Tensor>& Q, const std::shared_ptr<Tensor>& V, MaskType mask) override;
    };
  }
}
