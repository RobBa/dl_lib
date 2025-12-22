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

Tensor::Tensor(const Tensor& other) {
  this->device = other.device;
  this->dims = other.dims;
  this->type = other.type;

  auto size = other.dims.getTotalSize();
  allocValues(size, other.device);
}

/**
 * @brief Scalar multiplication, capable of broadcasting. First argument assumed
 * to be the scalar tensor.
 */
Tensor Tensor::multiply1D(const Tensor& scalar, const Tensor& right) const {
  Tensor res(right);
  for(int i=0; i<right.dims.getTotalSize(); ++i){
    res.values[i] = this->values[0] * right.values[i];
  }
  return res;
}

/**
 * @brief Just like in normal matrix multiplication order matters.
 * Not commutative -> a*b != b*a
 * 
 * We assume here that the dimensions of the tensors already do match!
 * The check of whether they do or not is to be performed by the surrounding
 * network class object instance upon construction. 
 */
Tensor Tensor::multiply2D(const Tensor& left, const Tensor& right) const {
  /* Tensor res(left.dims.get(0), right.dims.get(1), this->device);
  for(uint16_t row=0; row<left.dims.get(0); row++){
    const uint32_t rowOffset = row * left.dims.get(1);
    
    for(uint16_t col=0; col<right.dims.get(1); col++){
      for(uint16_t idx=0; idx<left.dims.get(1); idx++){
        const uint32_t leftOffset = rowOffset + idx;
        const uint32_t targetOffset =  
        res[]
      }
    }
  } */
}

Tensor Tensor::operator*(Tensor const& other) const {
  if(device==Device::CUDA){
    __throw_invalid_argument("Not implemented");
  }

  if(other.type == TensorType::OneD){
    multiply1D(other, *this);
  }

  switch(type){      
    case TensorType::OneD:
      return multiply1D(*this, other);
    case TensorType::TwoD:
      return multiply2D(*this, other);
    default:
      __throw_invalid_argument("Not implemented yet");
  }
}