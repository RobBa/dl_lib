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

#include "computational_graph/graph_node.h"
#include "computational_graph/topological_sort.h"

#include <utility>
#include <limits>
#include <cstring>

#include <format>

#ifdef __CUDA
#include "utility/cuda/cuda_common.cuh"
#include "data_modeling/cuda/tensorops.cuh"
#endif

using namespace std;

/******************************************************************** 
*************************** tensorValues_t *********************************
********************************************************************/

Tensor::tensorValues_t::tensorValues_t() {
  device = defaultDevice;
}

Tensor::tensorValues_t::tensorValues_t(Device d) : device(d) {}

Tensor::tensorValues_t::tensorValues_t(tensorValues_t&& other) noexcept 
  : device{std::move(other.device)}, size{std::move(other.size)}, values{std::move(other.values)} { }

Tensor::tensorValues_t& Tensor::tensorValues_t::operator=(tensorValues_t&& other) noexcept {
  if (this == &other) return *this;

  device = std::move(other.device);
  size = std::move(other.size);
  values = std::move(other.values);

  return *this;
}

Tensor::tensorValues_t::~tensorValues_t() noexcept {
  assert(values != nullptr);

  switch(device){
    case Device::CPU:
      free(values);
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaFree(values));
      #else
        std::__throw_invalid_argument("Not compiled with CUDA.");
      #endif
      break;
  }
}

void Tensor::tensorValues_t::resize(const tensorSize_t size) {
  this->size = size;
  switch (device) {
    case Device::CPU:
      values = static_cast<ftype*>(std::malloc(this->size * sizeof(ftype)));
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaMalloc((void**) &values, this->size * sizeof(ftype)));
      #else
        std::__throw_invalid_argument("Not compiled with CUDA.");
      #endif
      break;
  }
}

ftype* Tensor::tensorValues_t::getData() const noexcept {
#ifndef __CUDA
  static_assert(false, "Should only be callable with CUDA enabled");
#endif

  if(device==Device::CPU){
    __throw_runtime_error("Should only be called on CUDA tensor.");
  }
  return values;
}

/**
 * @brief Copy from pointer into this object.
 */
void Tensor::tensorValues_t::copyFromRaw(const ftype* src, tensorSize_t n) {
  assert(n == size);
  switch(device){
    case Device::CPU:
      std::memcpy(values, src, n * sizeof(ftype));
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaMemcpy(values, src, n*sizeof(ftype), cudaMemcpyDeviceToDevice));
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }
}

/**
 * @brief For convenience, since copy- and std::move-constructors and assigment operators
 * do not create a deepcopy, but construct another pointer pointing to the same piece
 * of memory.
 */
void Tensor::tensorValues_t::copyValues(Tensor::tensorValues_t& target) const {
  assert(device==target.device && size==target.size);

  switch(device){
    case Device::CPU:
      std::memcpy(target.values, values, size * sizeof(ftype));
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaMemcpy(target.values, values, size * sizeof(ftype), cudaMemcpyDeviceToDevice));
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }
}

/**
 * @brief Does what you think it does. For linear slicing.
 */
void Tensor::tensorValues_t::copyValues(tensorValues_t& target, tensorSize_t low, 
                                        tensorSize_t high, tensorSize_t targetOffset) const {
  assert(target.size >= high - low);
  if(high<low){
    __throw_invalid_argument(
      std::format("high argument should be higher than low, but received {0} and {1}", high, low).c_str()
    );
  }

  switch(device){
    case Device::CPU:
      std::memcpy(target.values+targetOffset, values+low, (high-low) * sizeof(ftype));
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaMemcpy(target.values+targetOffset, values+low, (high-low) * sizeof(ftype), cudaMemcpyDeviceToDevice));
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }
}

/**
 * @brief Indexed slicing along first dimension.
 * 
 * @param indices The indices of the first dimension.
 * @param sizeOfDim Complete size of the flattened first dimension.
 */
void Tensor::tensorValues_t::copyValues(tensorValues_t& target, span<const tensorDim_t> indices, 
                                        const tensorSize_t sizeOfDim) const {
  assert(target.size >= sizeOfDim * indices.size());

  switch(device){
    case Device::CPU: {
      tensorSize_t targetOffset = 0;
      for(tensorDim_t idx: indices){
        tensorSize_t thisOffset = idx * sizeOfDim;
        copyValues(target, thisOffset, thisOffset+sizeOfDim, targetOffset);
        targetOffset += sizeOfDim;
      }
      break; 
    }
    case Device::CUDA:
      #ifdef __CUDA
        // TODO: we can do streams here, that's why synchronize
        tensorSize_t targetOffset = 0;
        for(tensorDim_t idx: indices){
          tensorSize_t thisOffset = idx * sizeOfDim;
          copyValues(target, thisOffset, thisOffset+sizeOfDim, targetOffset);
          targetOffset += sizeOfDim;
        }
        cudaErrchk(cudaDeviceSynchronize());
      #else 
        __throw_invalid_argument("Not compiled with CUDA");
      #endif
      break;
  }
}

void Tensor::tensorValues_t::setDevice(const Device d) noexcept {
  #ifndef __CUDA
    cerr << "setDevice called when CUDA not enabled. Doing nothing." << endl;
    return;
  #else 
    if(d==device){
      return;
    }

    switch(device){
      case Device::CPU:{
        ftype* tmp;
        cudaErrchk(cudaMalloc((void**) &tmp, this->size * sizeof(ftype)));
        cudaErrchk(cudaMemcpy(tmp, values, this->size * sizeof(ftype), cudaMemcpyHostToDevice));
        free(values);
        values = tmp;
        break;
      }
      case Device::CUDA:{
        ftype* tmp = static_cast<ftype*>(std::malloc(this->size * sizeof(ftype)));
        cudaErrchk(cudaMemcpy(tmp, values, this->size * sizeof(ftype), cudaMemcpyDeviceToHost));
        cudaErrchk(cudaFree(values));
        values = tmp;
        break;
      }
    }

    device = d;
  #endif
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
      #ifdef __CUDA
        cuda::elementwiseadd(values, values, other.values, size);
      #else
        __throw_invalid_argument("Not compiled with CUDA");
      #endif
      break;
  }

  return *this;
}

ftype& Tensor::tensorValues_t::operator[](const tensorSize_t idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  if(device!=Device::CPU){
    __throw_invalid_argument("'ftype& operator[] const' only implemented for CPU");
  }
  return values[idx];
}

ftype Tensor::tensorValues_t::operator[](const tensorSize_t idx) const {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  switch(device){
    case Device::CPU:
      return values[idx];
    case Device::CUDA:
      #ifdef __CUDA
        return cuda::get(values, idx);
      #else
        __throw_invalid_argument("Not compiled with CUDA");
        break;
      #endif
  }

  __throw_runtime_error("Should never reach here.");
  return 0; // suppress warnings
}

void Tensor::tensorValues_t::set(ftype v, tensorSize_t idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  switch(device){
    case Device::CPU:
      values[idx] = v;
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cuda::set(v, values, idx);
      #else
        __throw_invalid_argument("Not compiled with CUDA");
      #endif
      break;
  }
}

ftype Tensor::tensorValues_t::get(tensorSize_t idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");
  
  switch(device){
    case Device::CPU:
      return values[idx];
    case Device::CUDA:
      #ifdef __CUDA
        return cuda::get(values, idx);
      #else
        __throw_invalid_argument("Not compiled with CUDA");
        break;
      #endif
  }

  __throw_runtime_error("Should never reach here.");
  return 0; // suppress warnings
}

/******************************************************************** 
*************************** Tensor **********************************
********************************************************************/

Tensor::Tensor(Tensor&& other) noexcept
  : dims{std::move(other.dims)}, 
    values{std::move(other.values)}, 
    requiresGrad{other.requiresGrad},
    cgNode{std::move(other.cgNode)}, 
    grads{std::move(other.grads)}
{ }

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  
  dims = std::move(other.dims);
  values = std::move(other.values);

  requiresGrad = other.requiresGrad;
  cgNode = std::move(other.cgNode);
  grads = std::move(other.grads);

  return *this;
}

/**
 * @brief Only createShallowCopy is supposed to access this ctor.
 * Like a copy ctor, shares recourses (grad, cgNode, values) with 
 * original (other) tensor.
 */
Tensor::Tensor(const Tensor& other, [[maybe_unused]] shallowCopyToken)
  : dims(other.dims), cgNode{other.cgNode}, values{other.values}, 
    grads{other.grads}, requiresGrad{other.requiresGrad} 
{
}

/**
 * @brief Creates an empty copy of this tensor.
 * Metadata all filled, but gradients not initialized, and 
 * values are reserved in memory, but uninitialized.
 */
Tensor Tensor::createEmptyCopy() const {
  auto res = Tensor(dims, values->getDevice(), requiresGrad);
  return res;
}

/**
 * @brief Creates a shallow copy. New tensor object, but all resources 
 * (grads, cgNode, values) are shared with this tensor. Useful for optimization.
 */
Tensor Tensor::createShallowCopy() const {
  auto res = Tensor(*this, shallowCopyToken{});
  return res;
}

/**
 * @brief Creates a linearized copy of this tensor. New memory allocations for 
 * values, and values will be copied so that the memory layout reflects the 
 * current shape of the tensor.
 */
Tensor Tensor::createLinearCopy() const {
  auto res = createEmptyCopy();
  
  switch(values->getDevice()){
    case Device::CPU:
    {
      for (tensorSize_t flatIdx = 0; flatIdx < values->getSize(); ++flatIdx) {
        tensorSize_t remainder = flatIdx;
        tensorSize_t srcOffset = 0;

        for (int i=dims.nDims()-1; i>=0; i--) {
          tensorSize_t coord = remainder % dims[i];
          remainder /= dims[i];
          srcOffset += coord * dims.getStride(i);
        }

        res.values->set((*values)[srcOffset], flatIdx);
      }
      break;
    }
    case Device::CUDA:
    {
      #ifdef __CUDA
        cuda::createLinearCopy(res, *this);
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
    }
  }


  return res;
}

/**
 * @brief Does a deep copy, but omits gradient and computational graph information.
 */
Tensor Tensor::createDeepCopy() const {
  assert(!grads || (grads && !grads->requiresGrad)); // gradient should not require gradient

  auto res = Tensor(dims, values->getDevice(), requiresGrad);
  values->copyValues(*res.values);

  assert(!res.grads); // TODO: check if this makes sense
  assert(!res.cgNode); // TODO: do we want to give it pointer of same node?
  return res;
}

/**
 * @brief For convenience. Needed for other classes to perform their CUDA operations.
 * Avoids moving all CUDA code into tensor.
 */
ftype* Tensor::getData() const noexcept {
  return values->getData();
}

/**
 * @brief Just like in normal matrix multiplication order matters.
 * Not commutative as per usual for matrices -> a*b != b*a
 * 
 * Multiply dimension left.dims.size()-1 of left with dimension right.dims.size()-2 of right.
 * Output shape: see document assumptions_matrices.md.
 * 
 * We assume here that the dimensions of the tensors already do match!
 * The check of whether they do or not is to be performed by the surrounding
 * network class object instance upon construction. 
 */
Tensor Tensor::matMulImpl(const Tensor& left, const Tensor& right) {
  if(left.dims.get(-1) != right.dims.get(-2)){
    __throw_runtime_error("Tensor dimensions do not match");
  }

  // broadcasting
  auto resDims = left.dims.nDims() > right.dims.nDims() ? left.dims.toVector() : right.dims.toVector();
  resDims[resDims.size()-2] = left.dims.get(-2); // rows
  resDims[resDims.size()-1] = right.dims.get(-1); // cols

  Tensor res(resDims, left.values->getDevice(), false);

  // sizes of the 2D matrices respectively
  const tensorSize_t leftSize = left.dims.get(-1) * left.dims.get(-2); 
  const tensorSize_t rightSize = right.dims.get(-1) * right.dims.get(-2);
  const tensorSize_t resSize = left.dims.get(-2) * right.dims.get(-1);

  switch(left.values->getDevice()){
    case Device::CPU:
    {
      tensorSize_t leftOffset = 0;
      tensorSize_t rightOffset = 0;
      tensorSize_t resOffset = 0;

      while(leftOffset < left.getSize()){
        matMul2DCpu(res, left, right, resOffset, leftOffset, rightOffset);
        leftOffset += leftSize;
        rightOffset += rightSize;
        resOffset += resSize;
      }

      break;
    }
    case Device::CUDA:
      #ifdef __CUDA
        cuda::matmul(res, left, right);
      #else
        __throw_invalid_argument("Not compiled with CUDA");
      #endif
      break;
  }

  return res;
}

/**
 * @brief Name says it all. Inplace operation on res.
 */
void Tensor::matMul2DCpu(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset, 
                           const tensorSize_t leftOffset, const tensorSize_t rightOffset) {
  
  const auto nRowsLeft = static_cast<tensorSize_t>(left.dims.get(-2));
  const auto nColsLeft = static_cast<tensorSize_t>(left.dims.get(-1));
  const auto nRowsRight = static_cast<tensorSize_t>(right.dims.get(-2));
  const auto nColsRight = static_cast<tensorSize_t>(right.dims.get(-1));

  for(tensorSize_t lrow=0; lrow<nRowsLeft; lrow++){
    const tensorSize_t resRowOffset = resOffset + lrow * nColsRight;
    const tensorSize_t leftRowOffset = leftOffset + lrow * nColsLeft;

    tensorSize_t rightIdx = rightOffset;
    // res likely has undefined memory content
    for(tensorSize_t rrow=0; rrow<nColsRight; rrow++){
      (*res.values)[resRowOffset+rrow] = (*left.values)[leftRowOffset] * (*right.values)[rightIdx];
      rightIdx++;
    }

    for(tensorSize_t lcol=1; lcol<nColsLeft; lcol++){
      const auto leftIdx = leftRowOffset + lcol;
      for(tensorSize_t rrow=0; rrow<nColsRight; rrow++){
        (*res.values)[resRowOffset+rrow] += (*left.values)[leftIdx] * (*right.values)[rightIdx];
        rightIdx++;
      }
    }
  }
}

/**
 * @brief Matrix multiplication.
 */
Tensor Tensor::matmul(const Tensor& other) const {
  assert(values->getDevice()==other.values->getDevice());

  if(values->getDevice()!=other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }
  return matMulImpl(*this, other);
}

/**
 * @brief Addition of two tensors. This works in two ways: 
 * 1. Shapes of the two tensors are identical. In this case it is simple 
 * elementwise addition.
 * 2. The second tensor is a vector. In this case broadcast it. We assume 
 * other.dims == (dimN) && this->dims == (dim0, dim1,..., dimN).
 */
Tensor Tensor::operator+(const Tensor& other) const {
  if(this->dims != other.dims && 
    !(other.dims.nDims() == 1 && other.dims.get(0) == dims.get(-1))){
    __throw_invalid_argument("Tensors need matching dimensions");
  }
  else if(values->getDevice()!=other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  Tensor res(dims, values->getDevice());
  switch(values->getDevice()){
    case Device::CPU:
      if(dims==other.dims){
        // elementwise add
        for(tensorSize_t i=0; i<values->getSize(); i++){
          (*res.values)[i] = (*values)[i] + (*other.values)[i];
        }
      }
      else [[likely]] { 
        // broadcasted add
        const auto stride = static_cast<tensorSize_t>(other.dims.get(0));
        for(tensorSize_t offset=0; offset<values->getSize(); offset+=stride){
          for(tensorSize_t i=0; i<stride; i++){
            (*res.values)[offset+i] = (*values)[offset+i] + (*other.values)[i];
          }
        }
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        if(dims==other.dims){
          cuda::elementwiseadd(res.getData(), values->getData(), other.getData(), values->getSize());
        }
        else [[likely]] {
          cuda::broadcastedadd(res.getData(), values->getData(), other.getData(), values->getSize());
        }
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }
  return res;
}

/**
 * @brief Named version of operator +.
 */
Tensor Tensor::add(const Tensor& other) const {
  return *this + other;
}

/**
 * @brief Elementwise multiplication.
 */
Tensor Tensor::operator*(const Tensor& other) const {
  if(this->dims != other.dims){
    __throw_invalid_argument("Tensors need same dimensions");
  }
  else if(values->getDevice()!=other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  Tensor res(dims, values->getDevice(), false);
  switch(values->getDevice()){
    case Device::CPU:
      for(tensorSize_t i=0; i<values->getSize(); i++){
        (*res.values)[i] = (*values)[i] * (*other.values)[i];
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cuda::elementwisemul(res.getData(), values->getData(), 
                             other.values->getData(), values->getSize());
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }

  return res;
}

/**
 * @brief Named version of operator *.
 */
Tensor Tensor::elementwiseMul(const Tensor& other) const {
  return *this * other;
}

Tensor Tensor::operator*(ftype scalar) const {
  Tensor res(dims, values->getDevice(), false);
  switch(values->getDevice()){
    case Device::CPU:
      for (tensorSize_t i = 0; i < values->getSize(); ++i) {
        (*res.values)[i] = (*values)[i] * scalar;
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cuda::scalarmul(res.getData(), values->getData(), scalar, values->getSize());
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }

  return res;
}

Tensor Tensor::operator/(ftype scalar) const {
  if(scalar==0.0){
    __throw_runtime_error("Cannot divide by zero.");
  }

  Tensor res(dims, values->getDevice(), false);
  switch(values->getDevice()){
    case Device::CPU:
      for (tensorSize_t i = 0; i < values->getSize(); ++i) {
        (*res.values)[i] = (*values)[i] / scalar;
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cuda::scalarmul(res.getData(), values->getData(), 1 / scalar, values->getSize());
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }

  return res;
}

Tensor Tensor::operator+(ftype scalar) const {
  Tensor res(dims, values->getDevice(), false);
  switch(values->getDevice()){
    case Device::CPU:
      for (tensorSize_t i = 0; i < values->getSize(); ++i) {
        (*res.values)[i] = (*values)[i] + scalar;
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cuda::scalaradd(res.getData(), values->getData(), scalar, values->getSize());
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }

  return res;
}

Tensor Tensor::operator-(ftype scalar) const {
  Tensor res(dims, values->getDevice(), false);
  switch(values->getDevice()){
    case Device::CPU:
      for (tensorSize_t i = 0; i < values->getSize(); ++i) {
        (*res.values)[i] = (*values)[i] - scalar;
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cuda::scalaradd(res.getData(), values->getData(), -scalar, values->getSize());
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }

  return res;
}

Tensor operator*(ftype scalar, const Tensor& tensor) {
  return tensor * scalar;
}

Tensor operator+(ftype scalar, const Tensor& tensor) {
  return tensor + scalar;
}

void Tensor::backward() {
  if(!requiresGrad){
    __throw_runtime_error("Invoking backward on Tensor with no grad");
  }
  else if(!cgNode){
    __throw_runtime_error("Invoking backward on Tensor not created by a differentiable operation");
  }

  // last node has no incoming gradients -> factor 1
  if (!grads) {
    grads = make_shared<Tensor>(dims, values->getDevice(), false);
    grads->reset(1);
  }

  vector<Tensor*> sortedTensors = cgraph::TopologicalSort::reverseSort(this);
  for(auto tPtr: sortedTensors){
    auto& tensor = *tPtr;
    assert(tensor.grads && !tensor.grads->requiresGrad); // gradient should not require grad

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

/**
 * @brief Get gradients
 */
shared_ptr<Tensor> Tensor::getGrads() const {
  return grads;
}

/**
 * @brief Sometimes we do accept negative dim-values. In accordance with e.g. 
 * NumPy we map from the end to the beginning in that case. 
 */
tensorDim_t Tensor::mapDim(const int dim, const Dimension& dims) {
  if(dim>=0){
    return dim;
  }
  else if(dim + dims.nDims() < 0){
    __throw_invalid_argument("Invalid dim value given.");
  }

  // dims < 0
  return dims.nDims() + dim;
}

void Tensor::transposeImplCpu(Tensor& target, const int dim1, const int dim2) const noexcept {
    assert(values->getSize() == target.values->getSize() && dims.nDims()==target.dims.nDims());
    if(dim1 == dim2) {
        return;
    }

    const auto& source = *this; // easier to read
    
    auto dim1Mapped = mapDim(dim1, source.dims);
    auto dim2Mapped = mapDim(dim2, source.dims);
    
    const int numDims = source.dims.nDims();
    std::vector<tensorSize_t> indices(numDims, 0);
    std::vector<tensorSize_t> dimSizes(numDims);
    std::vector<tensorSize_t> sourceStrides(numDims);
    
    // strides for source
    tensorSize_t stride = 1;
    for(int d = numDims - 1; d >= 0; d--) {
        dimSizes[d] = source.dims.get(d);
        sourceStrides[d] = stride;
        stride *= dimSizes[d];
    }
    
    // strides for target
    std::vector<tensorSize_t> targetDimSizes = dimSizes;
    std::swap(targetDimSizes[dim1Mapped], targetDimSizes[dim2Mapped]);
    
    std::vector<tensorSize_t> targetStrides(numDims);
    stride = 1;
    for(int d = numDims - 1; d >= 0; d--) {
        targetStrides[d] = stride;
        stride *= targetDimSizes[d];
    }

    auto transposedValues = make_unique<tensorValues_t>(source.values->getDevice());
    transposedValues->resize(source.values->getSize());

    tensorSize_t totalSize = source.values->getSize();
    for(tensorSize_t targetIdx = 0; targetIdx < totalSize; targetIdx++) {
        // linear target index to multi-dimensional idx
        tensorSize_t tmp = targetIdx;
        for(int d = 0; d < numDims; d++) {
            indices[d] = tmp / targetStrides[d];
            tmp %= targetStrides[d];
        }
        
        // Swap the transposed dimensions to get source indices
        std::swap(indices[dim1Mapped], indices[dim2Mapped]);
        
        // Convert multi-dimensional source indices to linear index
        tensorSize_t sourceIdx = 0;
        for(int d = 0; d < numDims; d++) {
            sourceIdx += indices[d] * sourceStrides[d];
        }
        
        (*transposedValues)[targetIdx] = (*source.values)[sourceIdx];
    }
    
    target.values = std::move(transposedValues);
    target.dims.swap(dim1Mapped, dim2Mapped);
}

/**
 * @brief Transposes 2D tensor. A little sleeker than transposeImplCpu.
 */
void Tensor::transposeImpl2DCpu(Tensor& target, const int dim1, const int dim2) const noexcept {
  assert(values->getSize()==target.values->getSize() && target.dims.nDims()==2 && dims.nDims()==2);
  if(dim1==dim2){
    return;
  }

  const auto& source = *this; // easier to read

  auto dim1Mapped = mapDim(dim1, source.dims);
  auto dim2Mapped = mapDim(dim2, source.dims);

  // large dim wraps small dim
  const auto largeDim = dim1Mapped < dim2Mapped ? dim1Mapped : dim2Mapped;
  const auto smallDim = dim1Mapped < dim2Mapped ? dim2Mapped : dim1Mapped;

  // largeDimSize >= smallDimSize
  const auto largeDimOffset = dims.getStride(largeDim);
  const auto smallDimOffset = dims.getStride(smallDim);

  auto transposedValues = make_unique<tensorValues_t>(source.values->getDevice());
  transposedValues->resize(source.values->getSize());

  tensorSize_t resIdx = 0;
  for(tensorSize_t smallDimCount=0; smallDimCount<source.dims.get(smallDim); smallDimCount++){
    for(tensorSize_t largeDimCount=0; largeDimCount<source.dims.get(largeDim); largeDimCount++){
      tensorSize_t offset = largeDimCount * largeDimOffset + smallDimCount * smallDimOffset;

      for(tensorSize_t smallDimIdx=0; smallDimIdx<smallDimOffset; smallDimIdx++){
        (*transposedValues)[resIdx] = (*source.values)[offset];
        resIdx++;
        offset++;
      }
    }
  }

  target.values = std::move(transposedValues);
  target.dims.swap(dim1Mapped, dim2Mapped);
}

/**
 * @brief Get a tensor that is linear in memory. Useful for coalesced memory access patterns.
 */
Tensor Tensor::getLinear() const {
  if(dims.inOriginalState())
    return createShallowCopy();
  return createLinearCopy();
}

/**
 * @brief Swap dim1 and dim2, modify this tensor.
 * 
 * Out of place operation.
 */
void Tensor::transposeThis(int dim1, int dim2) noexcept {
  switch(values->getDevice()){
    case Device::CPU:
      if(dims.nDims()>2){
        transposeImplCpu(*this, dim1, dim2);
      }
      else{
        transposeImpl2DCpu(*this, dim1, dim2);
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        if(dims.nDims()>2){
          // TODO: can make this better inplace?
          cuda::transpose(values->getData(), values->getData(), dims, dim1, dim2);
        }
        else{
          // TODO: can make this better inplace?
          cuda::transpose2D(values->getData(), values->getData(), dims, dim1, dim2);
        }
      #else 
        __throw_runtime_error("Not implemented with CUDA");
      #endif
      break;
  }
}

/**
 * @brief Out of place transposition of last two axes.
 * 
 */
void Tensor::transposeThis() noexcept {
  if(dims.nDims()<2){
    return;
  }
  transposeThis(-1, -2);
}

/**
 * @brief Like overloaded transpose with requiresGrad==false.
 */
Tensor Tensor::transpose(int dim1, int dim2) const {
  return transpose(dim1, dim2, false);
}

/**
 * @brief Like transposeThis, but returns a new tensor.
 * We give requiresGrad as an optional argument to give more control of 
 * what this tensor is intended to do. E.g. in backprop sometimes we do 
 * need to create transposed tensors to multiply with, and with 
 * requiresGrad==false we avoid unnecessary memory allocation overhead.
 */
Tensor Tensor::transpose(int dim1, int dim2, const bool requiresGrad) const {
  Tensor res(dims, values->getDevice(), requiresGrad);
  
  switch(values->getDevice()){
    case Device::CPU:
      if(dims.nDims()>2){
        transposeImplCpu(res, dim1, dim2);
      }
      else{
        transposeImpl2DCpu(res, dim1, dim2);
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        if(dims.nDims()>2){
          cuda::transpose(res.values->getData(), values->getData(), res.dims, dim1, dim2);
        }
        else{
          cuda::transpose2D(res.values->getData(), values->getData(), res.dims, dim1, dim2);
        }
      #else 
        __throw_runtime_error("Not implemented with CUDA");
      #endif
      break;
  }

  return res;
}

/**
 * @brief Reorder according to the newOrder. 
 * 
 * New order aligns axes newly. E.g. (2, 3, 1, 0)
 */
void Tensor::permute(const std::vector<tensorDim_t>&& newOrder) noexcept {
  // TODO: highly inefficient -> refactor
  assert(newOrder.size()<=std::numeric_limits<tensorDim_t>::max());
  for(tensorDim_t i=0; i<static_cast<tensorDim_t>(newOrder.size()); i++){
    transpose(i, newOrder[i]);
  }
}

/**
 * @brief Populates the tensor with value.
 */
void Tensor::reset(const ftype x) noexcept {
  switch(values->getDevice()){
    case Device::CPU:
      memset(values->getData(), x, values->getSize());
      break;
    case Device::CUDA:
      #ifdef __CUDA
        cudaErrchk(cudaMemset(values->getData(), x, values->getSize() * sizeof(ftype)));
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }
}

/**
 * @brief Populates the tensor with values drawn according to initializer.
 */
void Tensor::reset(const shared_ptr<utility::InitializerBase> init) noexcept {
  switch(values->getDevice()){
    case Device::CPU:
      for(tensorSize_t i=0; i<values->getSize(); i++){
        (*values)[i] = init->drawNumber();
      }
      break;
    case Device::CUDA:
      #ifdef __CUDA
        auto newValues = static_cast<ftype*>(std::malloc(values->getSize() * sizeof(ftype)));
        for(tensorSize_t i=0; i<values->getSize(); i++){
          newValues[i] = init->drawNumber();
        }
        cudaErrchk(cudaMemcpy(values->getData(), newValues, values->getSize() * sizeof(ftype), cudaMemcpyHostToDevice));
        free(newValues);
        // TODO: better initialize directly on GPU
      #else 
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
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
 * @brief Gets a slice of this tensor.
 * 
 * Quick and dirty implementation for now: Copies and
 * returns. 
 * 
 * @param low Lower idx, inclusive bound.
 * @param high Upper idx, non-inclusive bound.
 * @return Tensor The slices tensor.
 */
Tensor Tensor::getSlice(tensorSize_t low, tensorSize_t high) const {
  if(high<=low){
    __throw_invalid_argument("Upper bound most be larger than lower bound.");
  }

  auto resDims = dims.toVector();
  resDims[0] = high-low;
  Tensor res(std::move(resDims), values->getDevice(), false);
  values->copyValues(*res.values, low, high, 0);
  return res;
}

/**
 * @brief Like overload, but gets the slicing according to the 
 * indices given by the argument. Used e.g. in batch-size.
 * 
 * @param indices A list of indices
 * @return Tensor The result.
 */
Tensor Tensor::getSlice(span<const tensorDim_t> indices) const {
  assert(indices.size()>0);
  
  auto resDims = dims.toVector();
  resDims[0] = indices.size();

  Tensor res(std::move(resDims), values->getDevice(), false);
  values->copyValues(*res.values, indices, res.getDims().getStride(0));
  return res;
}

/**
 * @brief Prints only sample of up to 2D tensors.
 */
void printValuesCpu(std::ostream& os, const Tensor& t) {
  auto printVals = [&os](const Tensor& t){
    constexpr auto MAX_IDX = static_cast<tensorDim_t>(10);

    if(t.dims.nDims()==2){
      for(tensorDim_t i=0; i<min(MAX_IDX, t.dims.get(0)); i++){
        for(tensorDim_t j=0; j<min(MAX_IDX, t.dims.get(1)); j++){
          os << t.get({i, j}) << " ";
        }
        os << "\n";
      }
    }
    else{
      for(tensorDim_t i=0; i<min(MAX_IDX, static_cast<tensorDim_t>(t.values->getSize())); i++){
        os << (*t.values)[i] << " ";
      }
    }
  };

  printVals(t);
  if(t.grads){
    os << "\n\nGrads:\n";
    printVals(*t.grads);
  }
}

/**
 * @brief Print out the first few values of the flattened array.
 */
#ifdef __CUDA
void printValuesCuda(std::ostream& os, const Tensor& t) {
  __throw_logic_error("printValuesCuda should not be reachable when not compiled with CUDA");
  auto printVals = [&os](const Tensor& t){
    constexpr auto MAX_IDX = static_cast<tensorSize_t>(10);

    const auto maxIdx = min(MAX_IDX, t.values->getSize());
    auto tmp = static_cast<ftype*>(std::malloc(t.getSize() * sizeof(ftype)));
    cudaErrchk(cudaMemcpy(tmp, t.getData(), maxIdx*sizeof(ftype), cudaMemcpyDeviceToHost));

    for(tensorSize_t i=0; i<maxIdx; i++){
      os << tmp[i];
    }
    os << "\n\n";

    free(tmp);
  };

  printVals(t);
  if(t.grads){
    os << "\n\nGrads:\n";
    printVals(*t.grads);
  }
}
#endif

ostream& operator<<(ostream& os, const Tensor& t) noexcept {
  os << "Dims: " << t.getDims();
  os << "\nDevice: " << DeviceToString(t.values->getDevice());
  os << "\nrequiresGrad: " << t.requiresGrad << "\n\n";

  switch(t.values->getDevice()){
    case Device::CPU:
      printValuesCpu(os, t);
      break;
    case Device::CUDA:
      #ifdef __CUDA
        printValuesCuda(os, t);
      #else
        __throw_runtime_error("Not compiled with CUDA");
      #endif
      break;
  }

  return os;
}

/**
 * @brief Computes the 1D index from a set of indices. 
 * 
 * WARNING: Does not check for overflow.
 */
tensorSize_t Tensor::computeLinearIdx(const std::vector<tensorDim_t>&& idx, const Dimension& dims) {
  return computeLinearIdx(idx, dims);
}

/**
 * @brief Computes the 1D index from a set of indices. 
 * 
 * WARNING: Does not check for overflow.
 */
tensorSize_t Tensor::computeLinearIdx(const std::vector<tensorDim_t>& idx, const Dimension& dims) {
  if(idx.size()!=dims.nDims()) {
    __throw_invalid_argument("Number of idxs must match number of dimensions.");
  }
  else if(idx.size()==0){
    return 0;
  }

  tensorSize_t res = 0;
  for(tensorDim_t i=0; i<idx.size(); i++){
    res += idx[i] * dims.getStride(i);
  }
  return res;
}

/**
 * @brief No explanation needed.
 */
ftype Tensor::get(const std::vector<tensorDim_t>& idx) const {
  return (*values)[computeLinearIdx(idx, dims)]; 
}

/**
 * @brief Special getter, indexes the contained underlying array linearly.
 * Can lead to unexpected results in multidimensional tensors.
 */
ftype Tensor::get(tensorSize_t idx) const {
  return (*this)[idx];
}

/**
 * @brief For convenience.
 */
ftype Tensor::operator[](tensorSize_t idx) const {
  return (*values)[idx];
}


ftype Tensor::get(tensorDim_t idx0, tensorDim_t idx1) const {
  return get({idx0, idx1});
}

ftype Tensor::get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) const {
  return get({idx0, idx1, idx2});
}

ftype Tensor::get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) const {
  return get({idx0, idx1, idx2, idx3});
}

/**
 * @brief No explanation needed.
 */
void Tensor::set(ftype item, const std::vector<tensorDim_t>& idx) {
  values->set(item, computeLinearIdx(idx, dims));
}

/**
 * @brief Special setter, indexes the contained underlying array linearly.
 * Can lead to unexpected results in multidimensional tensors.
 */
void Tensor::set(ftype item, tensorDim_t idx) { 
  values->set(item, idx);
}

void Tensor::set(ftype item, tensorDim_t idx0, tensorDim_t idx1) { 
  set(item, {idx0, idx1});
}

void Tensor::set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) { 
  set(item, {idx0, idx1, idx2});
}

void Tensor::set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) { 
  set(item, {idx0, idx1, idx2, idx3});
}