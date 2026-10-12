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

#include "self_attention_base.h"

namespace module {
  namespace transformer {
    /** 
     * The base class for all the layers that we have. Not instantiable.
     */
    class DLLIB_API SdpAttention : public SelfAttentionBase {
      public:
        SdpAttention() = default;

        SdpAttention(const SdpAttention& other) = delete;
        SdpAttention& operator=(const SdpAttention& other) = delete;

        SdpAttention(SdpAttention&& other) noexcept = default;
        SdpAttention& operator=(SdpAttention&& other) noexcept = default;

        ~SdpAttention() noexcept = default;

        Tensor forward(const Tensor& K, const Tensor& Q, const Tensor& V, transformer::MaskEnum mask) override;
        std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& K, const std::shared_ptr<Tensor>& Q, const std::shared_ptr<Tensor>& V, MaskEnum mask) override;
    };
  }
}
