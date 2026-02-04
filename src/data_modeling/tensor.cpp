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

#include "graph_node.h"
#include "topological_sort.h"

#include <utility>

#ifndef NDEBUG
  #include <iostream>
#endif // NDEBUG

using namespace std;

/******************************************************************** 
*************************** tensorValues_t *********************************
********************************************************************/

Tensor::tensorValues_t::tensorValues_t() {
  device = defaultDevice;
}

Tensor::tensorValues_t::tensorValues_t(Device d) : device(d) {}

Tensor::tensorValues_t::tensorValues_t(tensorValues_t&& other) noexcept 
  : device{move(other.device)}, size{move(other.size)}, values{move(other.values)} { }

Tensor::tensorValues_t& Tensor::tensorValues_t::operator=(tensorValues_t&& other) noexcept {
  if (this == &other) return *this;

  device = move(other.device);
  size = move(other.size);
  values = move(other.values);

  return *this;
}

Tensor::tensorValues_t::~tensorValues_t() noexcept {
  assert(values != nullptr);

  switch(device){
    case Device::CPU:
      free(values);
      break;
    case Device::CUDA:
      std::__throw_invalid_argument("Cuda destructor not implemented yet.");
      break;
  }
}

/**
 * @brief For convenience, since copy- and move-constructors and assigment operators
 * do not create a deepcopy, but construct another pointer pointing to the same piece
 * of memory.
 */
void Tensor::tensorValues_t::copyValues(Tensor::tensorValues_t& target, 
                                            const Tensor::tensorValues_t& origin) {
  assert(origin.device==target.device && origin.size==target.size);

  switch(origin.device){
    case Device::CPU:
      for(tensorSize_t i=0; i<origin.size; i++){
        target[i] = origin.values[i];
      }
      break;
    case Device::CUDA:
      __throw_runtime_error("CUDA not implemented for deep copy");
  }
}

void Tensor::tensorValues_t::setDevice(const Device d) noexcept {
  device = d;
}

Device Tensor::tensorValues_t::getDevice() const noexcept {
  return device;
}

void Tensor::tensorValues_t::setDefaultDevice(const Device d) noexcept {
  defaultDevice = d;
}
            
Device Tensor::tensorValues_t::getDefaultDevice() noexcept {
  return defaultDevice;
}

tensorSize_t Tensor::tensorValues_t::getSize() const noexcept {
  return size;
}

Tensor::tensorValues_t::operator bool() const noexcept {
  return values != nullptr;
}

void Tensor::tensorValues_t::addOtherCpu(const Tensor::tensorValues_t& other) noexcept {
  for(tensorSize_t i=0; i<this->size; i++){
    this->values[i] += other.values[i];
  }
}

Tensor::tensorValues_t& 
Tensor::tensorValues_t::operator+=(const Tensor::tensorValues_t& other) {
  assert(this->size==other.size && this->device == other.device);
  
  switch(device) {
    case Device::CPU:
      addOtherCpu(other);
      break;
    case Device::CUDA:
      __throw_invalid_argument("CUDA not supported yet for += operation");
  }

  return *this;
}

ftype& Tensor::tensorValues_t::operator[](int idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  switch(device){
    case Device::CPU:
      return values[idx];
    case Device::CUDA:
      __throw_invalid_argument("Cuda operator[] not implemented");
  }

  __throw_invalid_argument("Unexpected device encountered");
  return values[0]; // never reached, suppress warning
}

ftype Tensor::tensorValues_t::get(const int idx) const {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  switch(device){
    case Device::CPU:
      return values[idx];
    case Device::CUDA:
      __throw_invalid_argument("Cuda getter not implemented");
  }

  __throw_invalid_argument("Unexpected device encountered");
  return 0; // never reached, suppress warning
}

/******************************************************************** 
*************************** Tensor **********************************
********************************************************************/

Tensor::Tensor(Tensor&& other) noexcept
  : dims{move(other.dims)}, type{other.type}, values{move(other.values)} 
{ }

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  
  this->type = other.type;
  this->dims = move(other.dims);
  this->values = move(other.values);

  return *this;
}

/**
 * @brief Creates an empty copy of this tensor.
 * Metadata all filled, but gradients not initialized, and 
 * values are reserved in memory, but uninitialized.
 */
Tensor Tensor::createEmptyCopy() const {
  return Tensor(values->getDevice(), dims);
}

Tensor Tensor::createDeepCopy() const {
  auto res = Tensor(values->getDevice(), dims);
  tensorValues_t::copyValues(*res.values, *this->values);
}


/**
 * @brief Scalar multiplication, capable of broadcasting. First argument assumed
 * to be the scalar tensor.
 */
Tensor Tensor::multiplyScalar(const Tensor& scalar, const Tensor& right) const {
  Tensor res(values->getDevice(), dims);
  for(int i=0; i<right.getSize(); ++i){
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
  if(left.dims.get(1) != right.dims.get(0)){
    __throw_runtime_error("Tensor dimensions do not match");
  }

  Tensor res(this->values->getDevice(), left.dims.get(0), right.dims.get(1));

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
Tensor Tensor::multiply(const Tensor& left, const Tensor& right) {
  if(left.getDims().get(0) != right.getDims().get(1)){
    // TODO: show meaningful message in python without exception
    __throw_invalid_argument("Dimensions don't match");
  }

  return left * right;
}

void Tensor::backward() {
  if(!requiresGrad){
    __throw_runtime_error("Invoking backward on Tensor with no grad");
  }

  // can this happen in first place?
  if (!grads) {
    grads.emplace(false, values->getDevice(), dims);
    for(tensorSize_t i=0; i<values->getSize(); i++){
      (*grads.value().values)[i] = 1;
    }
  }

  vector<Tensor*> sortedTensors = graph::TopologicalSort::reverseSort(this);
  for(auto tPtr: sortedTensors){
    auto& tensor = *tPtr;
    if(tensor.cgNode){
      auto incomingGrads = tensor.cgNode->backward(*tensor.grads);
    }
  }
}

/**
 * @brief Populates the tensor with value.
 */
void Tensor::reset(const ftype x) {
  for(tensorSize_t i=0; i<values->getSize(); i++){
    (*values)[i] = x;
  }
}

/**
 * @brief Populates the tensor with values drawn according to initializer.
 */
void Tensor::reset(const utility::InitClass ic) {
  const auto init = utility::InitializerFactory::getInitializer(ic);
  for(tensorSize_t i=0; i<values->getSize(); i++){
    (*values)[i] = init->drawNumber();
  }
}

const Dimension& Tensor::getDims() const noexcept {
  return dims;
}

tensorSize_t Tensor::getSize() const noexcept {
  return values->getSize();
}

void Tensor::setDefaultDevice(const Device d) noexcept {
  tensorValues_t::setDefaultDevice(d);
}
            
Device Tensor::getDefaultDevice() noexcept {
  return tensorValues_t::getDefaultDevice();
}

void Tensor::setDevice(const Device d) noexcept {
  values->setDevice(d);
}

Device Tensor::getDevice() const noexcept {
  return values->getDevice();
}

void printValuesCpu(std::ostream& os, const Tensor& t) {
  const auto& dims = t.getDims();
  const auto MAX_IDX = static_cast<tensorDim_t>(5);

#ifndef NDEBUG
  for(int i=0; i<4; i++){
    cout << "Dim " << i << ": " << dims.get(i) << endl;
  }
#endif // NDEBUG

  if(dims.get(3)>0){
    std::__throw_invalid_argument("Printing 4D tensor not implemented");
  }
  else if(dims.get(2)>0){
    std::__throw_invalid_argument("Printing 3D tensor not implemented");
  }
  else if(dims.get(1)>0){
    for(uint8_t i=0; i<min(MAX_IDX, dims.get(0)); i++){
      for(uint8_t j=0; j<min(MAX_IDX, dims.get(1)); j++){
        os << t.get(i, j) << " ";
      }
      os << "\n";
    }
  }
  else{
    for(uint8_t i=0; i<min(MAX_IDX, dims.get(0)); i++){
      os << t.get(i) << " ";
    }
  }
}

ostream& operator<<(ostream& os, const Tensor& t) noexcept {
  os << "Dims: " << t.getDims() << "\n";
  os << "Device: " << DeviceToString(t.values->getDevice()) << "\n";

  switch(t.values->getDevice()){
    case Device::CPU:
      printValuesCpu(os, t);
      break;
    case Device::CUDA:
      __throw_invalid_argument("CUDA not supported yet in printing");
      break;
  }

  return os;
}

ftype Tensor::get(const int idx) const {
  assert(type==TensorType::OneD);
  return values->get(idx);
}

ftype Tensor::get(const int idx1, const int idx2) const {
  assert(type==TensorType::TwoD);
  return values->get(idx1 * dims.get(1) + idx2);  
}

ftype Tensor::get(const int idx1, const int idx2, const int idx3) const {
  __throw_runtime_error("3D indexing not implemented yet");
}

ftype Tensor::get(const int idx, const int idx2, const int idx3, const int idx4) const {
  __throw_runtime_error("4D indexing not implemented yet");
}

void Tensor::set(ftype item, int idx) {
  assert(type==TensorType::OneD);
  (*values)[idx] = item;
}

void Tensor::set(ftype item, int idx1, int idx2) {
  assert(type==TensorType::TwoD);
  (*values)[idx1 * dims.get(1) + idx2] = item;  
}

void Tensor::set(ftype item, int idx1, int idx2, int idx3) {
  __throw_runtime_error("3D indexing not implemented yet");
}

void Tensor::set(ftype item, int idx, int idx2, int idx3, int idx4) {
  __throw_runtime_error("4D indexing not implemented yet");
}

ftype& Tensor::operator[](const tensorSize_t idx) {
  return (*values)[idx];
}