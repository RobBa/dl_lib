/**
 * @file attention_head.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "module/layers/ff_layer.h"

#include "module/transformer/attention/self_attention_fn_base.h"
#include "module/transformer/params/params.h"

#include <vector>
#include <memory>

namespace module {
  namespace transformer {
    class MultiheadSelfAttn final {
      private:
        class AttnHead {
          private:
            module::FfLayer K;
            module::FfLayer Q;
            module::FfLayer V;

            std::shared_ptr<SelfAttentionFnBase> attn;

          public:
            AttnHead(std::shared_ptr<SelfAttentionFnBase> attn, const tensorDim_t embeddingDim, const tensorDim_t headDim) 
              : attn{std::move(attn)}, 
                K{embeddingDim, headDim, true, true}, 
                Q{embeddingDim, headDim, true, true}, 
                V{embeddingDim, headDim, true, true}  {}

            Tensor operator()(const Tensor& input, MaskType mask);
            std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input, MaskType mask);
        };

        std::vector<AttnHead> heads;
        
      public:
        MultiheadSelfAttn(SelfAttentionType attnType, tensorDim_t nHeads, tensorDim_t embeddingDim, tensorDim_t hiddenDim);

        Tensor operator()(const Tensor& input, MaskType mask);
        std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input, MaskType mask);
    };
  }
}
