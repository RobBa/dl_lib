/**
 * @file self_attention_fn_base.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "data_modeling/tensor.h"
#include "module/transformer/params/params.h"

#include <memory>

#include "utility/utils.h"

namespace module {
  namespace transformer {
    /** 
     * Self attention layers meant to be interchangeable.
     */
    class SelfAttentionFnBase {
      public:
        SelfAttentionFnBase() = default;

        SelfAttentionFnBase(const SelfAttentionFnBase& other) = delete;
        SelfAttentionFnBase& operator=(const SelfAttentionFnBase& other) = delete;

        SelfAttentionFnBase(SelfAttentionFnBase&& other) noexcept = default;
        SelfAttentionFnBase& operator=(SelfAttentionFnBase&& other) noexcept = default;

        virtual ~SelfAttentionFnBase() noexcept = default;

        virtual Tensor operator()(const Tensor& K, const Tensor& Q, const Tensor& V, MaskType mask) = 0;
        virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& K, const std::shared_ptr<Tensor>& Q, const std::shared_ptr<Tensor>& V, MaskType mask) = 0;
    };
  }
}