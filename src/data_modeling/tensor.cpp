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
  Tensor res(left.dims.get(0), right.dims.get(1), this->device);

  for(uint16_t row=0; row<left.dims.get(0); row++){
    const uint32_t leftRowOffset = row * left.dims.get(1);
    const uint32_t resRowOffset = row * right.dims.get(1);
    
    // TODO: can we optimize mem-access for right matrix?
    for(uint16_t col=0; col<right.dims.get(1); col++){
      ftype scalarProd = 0;
      
      for(uint16_t idx=0; idx<left.dims.get(1); idx++){
        const uint32_t leftOffset = leftRowOffset + idx;
        const uint32_t targetOffset = col + idx * right.dims.get(1); // we can do this better via increments
        scalarProd += left.values[leftOffset] * right.values[targetOffset];
      }

      const uint32_t resOffset = resRowOffset + col; // we can do this better via increments
      res.values[resOffset] = scalarProd;
    }
  }

  return res;
}

/**
 * @brief Multiplies two matrices in unsafe manner. If dimensions don't match this can lead to 
 * segmentation faults or wrong results. For safe method use static function multiply().
 */
Tensor Tensor::operator*(const Tensor& other) const {
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

/**
 * @brief Because we try to save runtime, we omit a dimensionality check in 
 * operator*(), with the responsibility of the dimensionality check going to
 * the network. This is a safe version of that functionality. 
 */
Tensor multiply(const Tensor& left, const Tensor& right) {
  if(left.getDims().get(0) != right.getDims().get(1)){
    // TODO: show meaningful message in python without exception
    __throw_invalid_argument("Dimensions don't match");
  }

  return left * right;
}