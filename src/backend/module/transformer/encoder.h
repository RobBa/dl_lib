/**
 * @file encoder.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "module/module_base.h"

#include "module/transformer/attention/self_attention_fn_base.h"
#include "module/transformer/params/params.h"

#include "utility/utils.h"

#include <memory>

namespace module {
  namespace transformer {
    class DLLIB_API Encoder : public ModuleBase {
      private:
        std::unique_ptr<SelfAttentionFnBase> attn;

      public:
        Encoder(SelfAttentionType attnType);

        Encoder(const Encoder& other) = delete;
        Encoder& operator=(const Encoder& other) = delete;

        Encoder(Encoder&& other) noexcept = default;
        Encoder& operator=(Encoder&& other) noexcept = default;

        ~Encoder() noexcept = default;

        Tensor operator()(const Tensor& input) const override;
        std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) const override;
    };
  }
}