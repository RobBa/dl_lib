/**
 * @file self_attention_base.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "data_modeling/tensor.h"
#include "module/transformer/params/mask.h"

#include <memory>

#include "utility/utils.h"

namespace module {
  namespace transformer {
    /** 
     * The base class for all the layers that we have. Not instantiable.
     */
    class DLLIB_API SelfAttentionBase {
      public:
        SelfAttentionBase() = default;

        SelfAttentionBase(const SelfAttentionBase& other) = delete;
        SelfAttentionBase& operator=(const SelfAttentionBase& other) = delete;

        SelfAttentionBase(SelfAttentionBase&& other) noexcept = default;
        SelfAttentionBase& operator=(SelfAttentionBase&& other) noexcept = default;

        ~SelfAttentionBase() noexcept = default;

        virtual Tensor forward(const Tensor& K, const Tensor& Q, const Tensor& V, MaskEnum mask) = 0;
        virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& K, const std::shared_ptr<Tensor>& Q, const std::shared_ptr<Tensor>& V, MaskEnum mask) = 0;
    };
  }
}