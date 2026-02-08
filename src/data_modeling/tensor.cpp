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
#include "add_node.h"
#include "matmul_node.h"
#include "elementwise_mul_node.h"
#include "topological_sort.h"

#include <utility>

#ifndef NDEBUG
  #include "safe_arithmetics.h"
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
  : dims{move(other.dims)}, values{move(other.values)} 
{ }

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  
  dims = move(other.dims);
  values = move(other.values);

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
Tensor Tensor::multiplyScalar(const Tensor& scalar, const Tensor& right) const noexcept {
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
 * Multiply dimension left.dims.size()-1 of left with dimension right.dims.size()-2 of right.
 * Output shape will be (left.dim(0), left.dim(1), ..., left.dim(-2), right.dim(-1))
 * 
 * We assume here that the dimensions of the tensors already do match!
 * The check of whether they do or not is to be performed by the surrounding
 * network class object instance upon construction. 
 */
Tensor Tensor::matMulImpl(const Tensor& left, const Tensor& right) const {
  if(left.dims.get(-1) != right.dims.get(-2)){
    __throw_runtime_error("Tensor dimensions do not match");
  }

  if(abs(static_cast<int>(right.dims.nDims()) - static_cast<int>(left.dims.nDims())) > 1){
    auto str = "Tensor dimension assumptions violated. See file 'assumption_matrices.md'.";
    __throw_invalid_argument(str);
  }

  auto resDims = left.dims.nDims() > right.dims.nDims() ? left.dims.toVector() : right.dims.toVector();
  resDims[resDims.size()-1] = right.dims.get(-1);
  resDims[resDims.size()-2] = left.dims.get(-2);
  Tensor res(resDims, values->getDevice());

  for(size_t dimension = 0; dimension<resDims.size(); dimension++){
    
  }

  return res;
}

/**
 * @brief Name says it all. Inplace operation on res
 */
void Tensor::matMul2DCpu(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset, 
                           const tensorSize_t leftOffset, const tensorSize_t rightOffset) const {

  const auto nRowsLeft = static_cast<tensorSize_t>(left.dims.get(-2));
  const auto nColsLeft = static_cast<tensorSize_t>(left.dims.get(-2));
  const auto nRowsRight = static_cast<tensorSize_t>(right.dims.get(-1));
  const auto nColsRight = static_cast<tensorSize_t>(right.dims.get(-1));

  for(tensorSize_t row=0; row<nRowsLeft; row++){
    const tensorSize_t leftRowOffset = row * nColsLeft;
    const tensorSize_t resRowOffset = row * nColsRight;
    
    tensorSize_t resIdx = resOffset + resRowOffset;
    // TODO: can we optimize mem-access for right matrix?
    for(tensorSize_t col=0; col<nColsRight; col++){
      ftype scalarProd = 0;
      
      tensorSize_t leftIdx = leftOffset + leftRowOffset;
      tensorSize_t rightIdx = rightOffset + col;
      for(tensorSize_t idx=0; idx<nColsLeft; idx++){
        scalarProd += (*left.values)[leftIdx] * (*right.values)[rightIdx];
        
        leftIdx++;
        rightIdx += nColsRight;
      }

      (*res.values)[resIdx] = scalarProd;
    }
  }
}

/**
 * @brief Matrix multiplication.
 */
Tensor Tensor::operator*(const Tensor& other) const {
  if(values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Multiplication not implemented on CUDA");
  }

  if(values->getDevice()==other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  static auto createGraphNode = [this, &other](Tensor&& t) -> Tensor {
    if (this->requiresGrad || other.requiresGrad) {
        t.requiresGrad = true;
        t.cgNode = std::make_shared<graph::MatMulNode>(this, &other);
    }
    return t;
  };

  if(other.dims.getSize()==1){
    return createGraphNode(multiplyScalar(other, *this));
  }

  if(dims.getSize()==1)
    return createGraphNode(multiplyScalar(*this, other)); // TODO: check what to do about this gradient
  
  return createGraphNode(matMulImpl(*this, other));
}

/**
 * @brief Named version of operator*.
 */
Tensor Tensor::matMul(const Tensor& other) const {
  return *this * other;
}

/**
 * @brief Elementise addition.
 */
Tensor Tensor::operator+(const Tensor& other) const {
  if(values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Multiplication not implemented on CUDA");
  }

  if(this->dims != other.dims){
    __throw_invalid_argument("Tensors need same dimensions");
  }
  else if(values->getDevice()==other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  assert(values->getSize()==other.values->getSize());

  static auto createGraphNode = [this, &other](Tensor&& t) -> Tensor {
    if (this->requiresGrad || other.requiresGrad) {
        t.requiresGrad = true;
        t.cgNode = std::make_shared<graph::AddNode>(this, &other);
    }
    return t;
  };

  Tensor res(dims, values->getDevice(), requiresGrad || other.requiresGrad);
  for(tensorSize_t i=0; i<values->getSize(); i++){
    (*res.values)[i] = values->get(i) + other.values->get(i);
  }

  return res;
}

/**
 * @brief Named version of operator +.
 */
Tensor Tensor::add(const Tensor& other) const {
  return *this + other;
}

Tensor Tensor::elementwiseMul(const Tensor& other) const {
  if(values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Multiplication not implemented on CUDA");
  }

  if(this->dims != other.dims){
    __throw_invalid_argument("Tensors need same dimensions");
  }
  else if(values->getDevice()==other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  assert(values->getSize()==other.values->getSize());

  static auto createGraphNode = [this, &other](Tensor&& t) -> Tensor {
    if (this->requiresGrad || other.requiresGrad) {
        t.requiresGrad = true;
        t.cgNode = std::make_shared<graph::ElementwiseMulNode>(this, &other);
    }
    return t;
  };

  Tensor res(dims, values->getDevice(), requiresGrad || other.requiresGrad);
  for(tensorSize_t i=0; i<values->getSize(); i++){
    (*res.values)[i] = values->get(i) * other.values->get(i);
  }

  return res;
}


void Tensor::backward() {
  if(!requiresGrad){
    __throw_runtime_error("Invoking backward on Tensor with no grad");
  }

  // check this one out for sure
  if (!grads) {
    grads = make_unique<Tensor>(false, values->getDevice(), dims);
    for(tensorSize_t i=0; i<values->getSize(); i++){
      (*grads->values)[i] = 1;
    }
  }

  vector<Tensor*> sortedTensors = graph::TopologicalSort::reverseSort(this);
  for(auto tPtr: sortedTensors){
    auto& tensor = *tPtr;
    if(tensor.cgNode){
      auto incomingGrads = tensor.cgNode->backward(*tensor.grads);
      const auto& parents = tensor.cgNode->getParents();

      for(size_t i=0; i<parents.size(); i++){
        auto parent = parents[i];
        if(!parent->requiresGrad){
          continue;
        }
        else if(!parent->grads){
          parent->grads = incomingGrads[i];
        }
        else{
          *parent->grads->values += *incomingGrads[i]->values;
        }
      }
    }
  }
}

/**
 * @brief 2D transposition. 
 * 
 * Can we generalize this to a higher degree without having special implementations
 * for each type of tensor? 
 */
void Tensor::transpose2D(Tensor& t) noexcept {
  if(values->getDevice()==Device::CUDA){
    __throw_runtime_error("2D transposition not implemented for CUDA");
  }
  
  const auto offset0 = dims.get(0);
  const auto offset1 = dims.get(1);
}

/**
 * @brief In place transpose.
 * 
 */
void Tensor::transpose() noexcept {
  // TODO: transpose
  transpose2D(*this);
  if(grads){ // TODO: does this make sense?
    transpose2D(*grads);
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

/**
 * @brief Prints only sample of up to 2D tensors.
 */
void printValuesCpu(std::ostream& os, const Tensor& t) {
  const auto& dims = t.getDims();
  const auto MAX_IDX = static_cast<tensorDim_t>(5);

#ifndef NDEBUG
  for(int i=0; i<dims.nDims(); i++){
    cout << "Dim " << i << ": " << dims.get(i) << endl;
  }
#endif // NDEBUG

  if(dims.nDims()>1){
    for(uint8_t i=0; i<min(MAX_IDX, dims.get(0)); i++){
      for(uint8_t j=0; j<min(MAX_IDX, dims.get(1)); j++){
        os << t.get({i, j}) << " ";
      }
      os << "\n";
    }
  }
  else{
    for(uint8_t i=0; i<min(MAX_IDX, dims.get(0)); i++){
      os << t.get({i}) << " ";
    }
  }
}

ostream& operator<<(ostream& os, const Tensor& t) noexcept {
  os << "\nDims: " << t.getDims() << "\n";
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

/**
 * @brief Computes the 1D index from a set of indices. 
 * 
 * WARNING: Does not check for overflow in release build.
 */
tensorSize_t Tensor::computeIdx(const std::vector<tensorDim_t>& idx) const {
  if(idx.size()!=dims.nDims()) {
    __throw_invalid_argument("Number of idxs should match number of dimensions.");
  }
  else if(idx.size()==0){
    return 1;
  }

  auto lastIdx = idx.size()-1;
#ifndef NDEBUG
  tensorSize_t res = idx[lastIdx];
#else 
  SafeArithmetics_t<tensorSize_t> res(idx[lastIdx]);
#endif // NDEBUG

  tensorSize_t offsetFactor = dims.get(lastIdx);
  for(size_t i=idx.size()-2; 0<=i; i--){
#ifndef NDEBUG
    res += idx[i] * offsetFactor;
#else
    res += SafeArithmetics_t<tensorSize_t>(idx[i]) * offsetFactor;
#endif // NDEBUG
    offsetFactor *= dims.get(i);
  }

#ifndef NDEBUG
  return res;
#else
  return res.value;
#endif // NDEBUG
}

/**
 * @brief No explanation needed.
 */
ftype Tensor::get(const std::vector<tensorDim_t>&& idx) const {
  return values->get(computeIdx(idx)); 
}

/**
 * @brief No explanation needed.
 */
void Tensor::set(ftype item, const std::vector<tensorDim_t>&& idx) {
  (*values)[computeIdx(idx)] = item;
}