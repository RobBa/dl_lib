/**
 * @file multihead_self_attention.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "multihead_self_attention.h"

#include "module/transformer/attention/sdp_attention_fn.h"

#include <stdexcept>

using namespace std;
using namespace module;
using namespace transformer;

/**
 * @brief Forward of individual head.
 */
Tensor MultiheadSelfAttn::AttnHead::operator()(const Tensor& input, const MaskType mask) {
  return (*attn)(K(input), Q(input), V(input), mask);
}

/**
 * @brief Forward of individual head.
 */
shared_ptr<Tensor> MultiheadSelfAttn::AttnHead::operator()(const shared_ptr<Tensor> input, const MaskType mask) {
  return (*attn)(K(input), Q(input), V(input), mask);
}

/**
 * @brief Construct a new Multihead Self Attn:: Multihead Self Attn object
 */
MultiheadSelfAttn::MultiheadSelfAttn(SelfAttentionType attnType, tensorDim_t nHeads, tensorDim_t embeddingDim, tensorDim_t hiddenDim) {
  if(hiddenDim % nHeads != 0) [[unlikely]] {
    __throw_invalid_argument("hiddenDim must be cleanly divisible by nHead s.t. (hiddenDim % nHeads) == 0");
  }

  shared_ptr<SelfAttentionFnBase> attn;
  switch(attnType) {
    case SelfAttentionType::ScaledDotProduct:
      attn = make_unique<SdpAttentionFn>();
      break;
    default:
      __throw_invalid_argument("Type of attention not implemented");
      attn = nullptr;
  }

  auto embedDim = hiddenDim / nHeads;
  for(int i = 0; i < nHeads; i++) {
    heads.emplace_back(attn, embedDim, hiddenDim);
  }
}

/**
 * @brief Forward of all heads, returns concatenated output.
 */
Tensor MultiheadSelfAttn::operator()(const Tensor& input, MaskType mask) {
  
}

/**
 * @brief Forward of all heads, returns concatenated output.
 */
std::shared_ptr<Tensor> MultiheadSelfAttn::operator()(std::shared_ptr<Tensor> input, MaskType mask) {

}