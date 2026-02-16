/**
 * @file relu.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "relu.h"
#include "global_params.h"

using namespace activation;

Tensor ReLU::operator()(Tensor& t) const noexcept {
  for(tensorSize_t i=0; i<t.getSize(); i++){
    constexpr ftype zero = 0;
    auto& target = t[i];
    if(zero > target){
      t[i] = 0;
    }
  }
  return t;
}

Tensor ReLU::gradient(const Tensor& t) noexcept {
/*   for(tensorSize_t i=0; i<t.getSize(); i++){
    constexpr ftype zero = 0;
    auto& target = t[i];
    if(zero > target){
      t[i] = 0;
    }
  }
  return t; */
}