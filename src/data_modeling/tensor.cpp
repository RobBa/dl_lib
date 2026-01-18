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

#include <utility>

#ifndef NDEBUG
  #include <iostream>
#endif // NDEBUG

using namespace std;

/******************************************************************** 
*************************** value_t *********************************
********************************************************************/

Tensor::value_t::value_t(Device d) : device(d) {}

Tensor::value_t::value_t(value_t&& other) noexcept {
  this->device = other.device;
  this->values = other.values;
}

Tensor::value_t& Tensor::value_t::operator=(value_t&& other) noexcept {
  if (this == &other) return *this;

  this->device = other.device;
  this->values = std::move(other.values);

  return *this;
}

Tensor::value_t::~value_t() noexcept {
  #ifndef NDEBUG
    if(values != nullptr){
      cerr << "value is nullptr in destructor of Tensor::value_t" << endl;
    }
  #endif // NDEBUG

  free(values);
}

/**
 * @brief For convenience, since copy- and move-constructors and assigment operators
 * do not create a deepcopy, but construct another pointer pointing to the same piece
 * of memory.
 */
Tensor::value_t Tensor::value_t::createDeepCopy(const Tensor::value_t& other) {
  value_t res(other.device);
  res.resize<tensorSize_t>(other.size);
  for(tensorSize_t i=0; i<other.size; i++){
    res[i] = other.values[i];
  }
  return res;
}

void Tensor::value_t::setDevice(const Device d) noexcept {
  this->device = d;
}

Device Tensor::value_t::getDevice() const noexcept {
  return this->device;
}

Tensor::value_t::operator bool() const noexcept {
  return values != nullptr;
}

ftype& Tensor::value_t::operator[](int idx) {
  return values[idx];
}


/******************************************************************** 
*************************** Tensor **********************************
********************************************************************/

Tensor::Tensor(const Tensor& other) noexcept {
  this->dims = other.dims;
  this->type = other.type;
  this->values = other.values;
}

Tensor& Tensor::operator=(const Tensor& other) noexcept {
  if (this == &other) return *this;
  
  this->dims = other.dims;
  this->type = other.type;
  this->values = other.values;

  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept {  
  this->type = other.type;
  this->dims = move(other.dims);
  this->values = move(other.values);
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  
  this->type = other.type;
  this->dims = move(other.dims);
  this->values = move(other.values);

  return *this;
}

/**
 * @brief Scalar multiplication, capable of broadcasting. First argument assumed
 * to be the scalar tensor.
 */
Tensor Tensor::multiplyScalar(const Tensor& scalar, const Tensor& right) const {
  Tensor res(right);
  for(int i=0; i<right.dims.getTotalSize(); ++i){
    (*res.values)[i] = (*this->values)[0] * (*right.values)[i];
  }
  return res;
}

/**
 * @brief Just like in normal matrix multiplication order matters.
 * Not commutative as per usual for matrices -> a*b != b*a
 * 
 * We assume here that the dimensions of the tensors already do match!
 * The check of whether they do or not is to be performed by the surrounding
 * network class object instance upon construction. 
 */
Tensor Tensor::multiply2D(const Tensor& left, const Tensor& right) const {
  Tensor res(left.dims.get(0), right.dims.get(1), this->values->getDevice());

  for(uint16_t row=0; row<left.dims.get(0); row++){
    const uint32_t leftRowOffset = row * left.dims.get(1);
    const uint32_t resRowOffset = row * right.dims.get(1);
    
    // TODO: can we optimize mem-access for right matrix?
    for(uint16_t col=0; col<right.dims.get(1); col++){
      ftype scalarProd = 0;
      
      for(uint16_t idx=0; idx<left.dims.get(1); idx++){
        const uint32_t leftOffset = leftRowOffset + idx;
        const uint32_t targetOffset = col + idx * right.dims.get(1); // we can do this better via increments
        scalarProd += (*left.values)[leftOffset] * (*right.values)[targetOffset];
      }

      const uint32_t resOffset = resRowOffset + col; // we can do this better via increments
      (*res.values)[resOffset] = scalarProd;
    }
  }

  return res;
}

/**
 * @brief Multiplies two matrices in unsafe manner. If dimensions don't match this can lead to 
 * segmentation faults or wrong results. For safe method use static function multiply().
 */
Tensor Tensor::operator*(const Tensor& other) const {
  if(this->values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Not implemented");
  }

  if(other.type == TensorType::OneD){
    multiplyScalar(other, *this);
  }

  switch(type){      
    case TensorType::OneD:
      return multiplyScalar(*this, other);
    case TensorType::TwoD:
      return multiply2D(*this, other);
    default:
      __throw_invalid_argument("Multiplication of tensors of higher order than 2 not implemented");
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

const Dimension& Tensor::getDims() const noexcept {
  return dims;
}