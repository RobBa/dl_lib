/**
 * @file tensor.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "tensor.h"

using namespace std;

const Dimension& Tensor::getDims() const noexcept {
  return dims;
}

Tensor::~Tensor() noexcept {
  if(values != nullptr){
    free(values);
  }
}

Tensor operator*(Tensor const& other){
  
}