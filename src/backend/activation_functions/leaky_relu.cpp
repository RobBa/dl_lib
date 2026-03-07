/**
 * @file leaky_relu.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "leaky_relu.h"

using namespace activation;

Tensor LeakyReLu::operator()(const Tensor& t) const noexcept {
  auto res = t.createDeepCopy();

  for(tensorSize_t i=0; i<t.getSize(); i++){
    constexpr ftype zero = 0;
    if(t[i] < zero){
      res.setItem(eps, i);
    }
  }

  return res;
}