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
      std::__throw_invalid_argument("Cuda destructor not implemented yet.");
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
      for(tensorSize_t i=0; i<size; i++){
        target[i] = values[i];
      }
      break;
    case Device::CUDA:
      __throw_runtime_error("CUDA not implemented for deep copy");
      break;
  }
}

/**
 * @brief Does what you think it does. For linear slicing.
 */
void Tensor::tensorValues_t::copyValues(tensorValues_t& target, tensorSize_t low, 
                                        tensorSize_t high, tensorSize_t targetOffset) const {
  assert(target.size >= high - low);

  switch(device){
    case Device::CPU:
      for(tensorSize_t i=0; i<high-low; i++){
        target[i] = values[low+i];
      }
      break;
    case Device::CUDA:
      __throw_runtime_error("CUDA not implemented for deep copy");
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
      __throw_runtime_error("CUDA not implemented for deep copy");
      break;
  }
}

void Tensor::tensorValues_t::setDevice(const Device d) noexcept {
  __throw_runtime_error("Not implemented yet.");
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
      break;
  }

  return *this;
}

ftype& Tensor::tensorValues_t::operator[](const tensorSize_t idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  if(device!=Device::CPU){
    __throw_invalid_argument("Operator [] only implemented for CPU");
  }
  return values[idx];
}

ftype Tensor::tensorValues_t::operator[](const tensorSize_t idx) const {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  if(device!=Device::CPU){
    __throw_invalid_argument("Operator [] only implemented for CPU");
  }
  return values[idx];
}

void Tensor::tensorValues_t::setItem(ftype v, tensorSize_t idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");

  switch(device){
    case Device::CPU:
      values[idx] = v;
      return;
    case Device::CUDA:
      __throw_runtime_error("Not implemented for CUDA yet");
  }

  __throw_runtime_error("Should never reach here.");
}

ftype Tensor::tensorValues_t::getItem(tensorSize_t idx) {
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");
  
  switch(device){
    case Device::CPU:
      return values[idx];
    case Device::CUDA:
      __throw_runtime_error("Not implemented for CUDA yet");
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
 * @brief Creates an empty copy of this tensor.
 * Metadata all filled, but gradients not initialized, and 
 * values are reserved in memory, but uninitialized.
 */
Tensor Tensor::createEmptyCopy() const {
  auto res = Tensor(dims, values->getDevice(), requiresGrad);
  return res;
}
/**
 * @brief Does a deep copy.
 */
Tensor Tensor::createDeepCopy() const {
  assert(!grads || (grads && !grads->requiresGrad)); // gradient should not require gradient

  auto res = Tensor(dims, values->getDevice(), requiresGrad);
  values->copyValues(*res.values);

  /* if(grads){
    res.grads = make_shared<Tensor>( grads->createDeepCopy() ); // TODO: do we want this?
  } */

  assert(!res.grads); // TODO: check if this makes sense
  assert(!res.cgNode); // TODO: do we want to give it pointer of same node?
  return res;
}


/**
 * @brief Scalar multiplication, capable of broadcasting. First argument assumed
 * to be the scalar tensor.
 */
Tensor Tensor::multiplyScalar(const Tensor& scalar, const Tensor& right) noexcept {
  Tensor res(right.dims, right.values->getDevice(), false);
  for(int i=0; i<right.getSize(); ++i){
    (*res.values)[i] = (*scalar.values)[0] * (*right.values)[i];
  }
  return res;
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
Tensor Tensor::matMulImpl(const Tensor& left, const Tensor& right) const {
  if(left.dims.getItem(-1) != right.dims.getItem(-2)){
    __throw_runtime_error("Tensor dimensions do not match");
  }

  if(abs(static_cast<int>(right.dims.nDims()) - static_cast<int>(left.dims.nDims())) > 1){
    auto str = "Tensor dimension assumptions violated. See file 'assumption_matrices.md'.";
    __throw_invalid_argument(str);
  }

  auto resDims = left.dims.nDims() > right.dims.nDims() ? left.dims.toVector() : right.dims.toVector();
  resDims[resDims.size()-2] = left.dims.getItem(-2); // rows
  resDims[resDims.size()-1] = right.dims.getItem(-1); // cols

  Tensor res(resDims, values->getDevice(), false);

  // sizes of the 2D matrices respectively
  const tensorSize_t leftSize = left.dims.getItem(-1) * left.dims.getItem(-2); 
  const tensorSize_t rightSize = right.dims.getItem(-1) * right.dims.getItem(-2);
  const tensorSize_t resSize = left.dims.getItem(-2) * right.dims.getItem(-1);

  tensorSize_t leftOffset = 0;
  tensorSize_t rightOffset = 0;
  tensorSize_t resOffset = 0;

  // lambda expected to get inlined by compiler
  auto multiplyNTimes = [&](const tensorDim_t n){
    for(tensorDim_t i=0; i<n; i++){
      matMul2DCpu(res, left, right, resOffset, leftOffset, rightOffset);

      leftOffset += leftSize;
      rightOffset += rightSize;
      resOffset += resSize;
    }
  };

  if(left.dims.nDims() == right.dims.nDims()){
    const auto nMultiplications = res.values->getSize() / resSize; // total size / size of 2D matrix
    multiplyNTimes(nMultiplications);
  }
  else if(left.dims.nDims() > right.dims.nDims()) {
    const auto nBatches = left.dims.getItem(0);

    for(tensorDim_t batch = 0; batch < nBatches; batch++){
      const auto nMultsPerBatch = res.values->getSize() / (nBatches * resSize);
      multiplyNTimes(nMultsPerBatch);
      rightOffset = 0;
    }
  }
  else {
    const auto nBatches = right.dims.getItem(0);

    for(tensorDim_t batch = 0; batch < nBatches; batch++){
      const auto nMultsPerBatch = res.values->getSize() / (nBatches * resSize);  
      multiplyNTimes(nMultsPerBatch);
      leftOffset = 0;
    }
  }

  return res;
}

/**
 * @brief Name says it all. Inplace operation on res
 */
void Tensor::matMul2DCpu(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset, 
                           const tensorSize_t leftOffset, const tensorSize_t rightOffset) {

  const auto nRowsLeft = static_cast<tensorSize_t>(left.dims.getItem(-2));
  const auto nColsLeft = static_cast<tensorSize_t>(left.dims.getItem(-1));
  const auto nRowsRight = static_cast<tensorSize_t>(right.dims.getItem(-2));
  const auto nColsRight = static_cast<tensorSize_t>(right.dims.getItem(-1));

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
      resIdx++;
    }
  }
}

/**
 * @brief Matrix multiplication.
 */
Tensor Tensor::matmul(const Tensor& other) const {
  assert(values->getDevice()==other.values->getDevice());
  if(values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Multiplication not implemented on CUDA");
  }

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
  if(values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Addition not implemented on CUDA");
  }

  if(this->dims != other.dims && 
    !(other.dims.nDims() == 1 && other.dims.getItem(0) == dims.getItem(-1))){
    __throw_invalid_argument("Tensors need matching dimensions");
  }
  else if(values->getDevice()!=other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  Tensor res(dims, values->getDevice());

  if(dims==other.dims){
    // elementwise add
    for(tensorSize_t i=0; i<values->getSize(); i++){
      (*res.values)[i] = (*values)[i] + (*other.values)[i];
    }
  }
  else { [[likely]]
    // broadcasted add
    const auto stride = static_cast<tensorSize_t>(other.dims.getItem(0));
    for(tensorSize_t offset=0; offset<values->getSize(); offset+=stride){
      for(tensorSize_t i=0; i<stride; i++){
        (*res.values)[offset+i] = (*values)[offset+i] + (*other.values)[i];
      }
    }
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
  assert(values->getDevice()==other.values->getDevice());
  if(values->getDevice()==Device::CUDA){
    __throw_invalid_argument("Multiplication not implemented on CUDA");
  }

  // TODO: check what to do about these two gradients and if you want broadcasting here at all
/*   if(other.dims.getSize()==1){
    return multiplyScalar(other, *this);
  }
  else if(dims.getSize()==1){
    return multiplyScalar(*this, other);
  } */

  if(this->dims != other.dims){
    __throw_invalid_argument("Tensors need same dimensions");
  }
  else if(values->getDevice()!=other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }

  assert(values->getSize()==other.values->getSize());
  Tensor res(dims, values->getDevice(), false);
  for(tensorSize_t i=0; i<values->getSize(); i++){
    (*res.values)[i] = (*values)[i] * (*other.values)[i];
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
  for (tensorSize_t i = 0; i < values->getSize(); ++i) {
    (*res.values)[i] = (*values)[i] * scalar;
  }

  return res;
}

Tensor Tensor::operator/(ftype scalar) const {
  if(scalar==0.0){
    __throw_runtime_error("Cannot divide by zero.");
  }

  Tensor res(dims, values->getDevice(), false);
  for (tensorSize_t i = 0; i < values->getSize(); ++i) {
    (*res.values)[i] = (*values)[i] / scalar;
  }

  return res;
}

Tensor Tensor::operator+(ftype scalar) const {
  Tensor res(dims, values->getDevice(), false);
  for (tensorSize_t i = 0; i < values->getSize(); ++i) {
    (*res.values)[i] = (*values)[i] + scalar;
  }

  return res;
}

Tensor Tensor::operator-(ftype scalar) const {
  Tensor res(dims, values->getDevice(), false);
  for (tensorSize_t i = 0; i < values->getSize(); ++i) {
    (*res.values)[i] = (*values)[i] - scalar;
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

  // check this one out for sure
  if (!grads) {
    grads = make_unique<Tensor>(dims, values->getDevice(), false);
    for(tensorSize_t i=0; i<values->getSize(); i++){
      (*grads->values)[i] = 1;
    }
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
  if(!grads){
    __throw_runtime_error("Tensor has no gradients.");
  }
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

void Tensor::transposeImpl(Tensor& target, const int dim1, const int dim2) const noexcept {
    assert(values->getSize() == target.values->getSize() && dims.nDims()==target.dims.nDims());
    if(target.getDevice() == Device::CUDA) {
        __throw_runtime_error("Transposition for CUDA not implemented yet");
    }
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
        dimSizes[d] = source.dims.getItem(d);
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
 * @brief Transposes 2D tensor. A little sleeker than transposeImpl.
 */
void Tensor::transposeImpl2D(Tensor& target, const int dim1, const int dim2) const noexcept {
  assert(values->getSize()==target.values->getSize() && target.dims.nDims()==2 && dims.nDims()==2);
  if(target.getDevice()==Device::CUDA){
    __throw_runtime_error("Transposition for CUDA not implemented yet");
  }
  else if(dim1==dim2){
    return;
  }

  const auto& source = *this; // easier to read

  auto dim1Mapped = mapDim(dim1, source.dims);
  auto dim2Mapped = mapDim(dim2, source.dims);

  // large dim wraps small dim
  const auto largeDim = dim1Mapped < dim2Mapped ? dim1Mapped : dim2Mapped;
  const auto smallDim = dim1Mapped < dim2Mapped ? dim2Mapped : dim1Mapped;

  // largeDimSize >= smallDimSize
  const auto largeDimOffset = getDimOffset(largeDim, dims);
  const auto smallDimOffset = getDimOffset(smallDim, dims);

  auto transposedValues = make_unique<tensorValues_t>(source.values->getDevice());
  transposedValues->resize(source.values->getSize());

  tensorSize_t resIdx = 0;
  for(tensorSize_t smallDimCount=0; smallDimCount<source.dims.getItem(smallDim); smallDimCount++){
    for(tensorSize_t largeDimCount=0; largeDimCount<source.dims.getItem(largeDim); largeDimCount++){
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

  /* if(source.grads){
    source.grads->transposeImpl(*target.grads, dim1, dim2); // TODO: do we need this?
  } */
}


/**
 * @brief Swap dim1 and dim2, modify this tensor.
 * 
 * Out of place operation.
 */
void Tensor::transposeThis(int dim1, int dim2) noexcept {
  if(dims.nDims()>2){
    transposeImpl(*this, dim1, dim2);
  }
  else{
    transposeImpl2D(*this, dim1, dim2);
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
  
  if(dims.nDims()>2){
    transposeImpl(res, dim1, dim2);
  }
  else{
    transposeImpl2D(res, dim1, dim2);
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
  values->copyValues(*res.values, indices, getDimOffset(0, resDims));
  return res;
}

/**
 * @brief Prints only sample of up to 2D tensors.
 */
void printValuesCpu(std::ostream& os, const Tensor& t) {
  auto printVals = [&os](const Tensor& t){
    constexpr auto MAX_IDX = static_cast<tensorDim_t>(10);

    if(t.dims.nDims()==2){
      for(tensorDim_t i=0; i<min(MAX_IDX, t.dims.getItem(0)); i++){
        for(tensorDim_t j=0; j<min(MAX_IDX, t.dims.getItem(1)); j++){
          os << t.getItem({i, j}) << " ";
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
    os << "Grads:\n";
    printVals(*t.grads);
  }
}

ostream& operator<<(ostream& os, const Tensor& t) noexcept {
  os << "Dims: " << t.getDims();
  os << "\nDevice: " << DeviceToString(t.values->getDevice());
  os << "\nrequiresGrad: " << t.requiresGrad << "\n\n";

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
    return 0; // TODO: this was 1. What is going on here?
  }

  const auto lastIdx = idx.size()-1;
  tensorSize_t offsetFactor = dims.getItem(lastIdx);
  
  tensorSize_t res = idx[lastIdx];
  for(int i=lastIdx-1; i>=0; i--){
    res += idx[i] * offsetFactor;
    offsetFactor *= dims.getItem(i);
  }

  return res;
}

/**
 * @brief Gets the total size of a dimension. E.g. if dims=(2, 3, 4),
 * the offset of dim1 is 3*4==12, and that of dim0 is 2*3*4==24.
 */
tensorSize_t Tensor::getDimOffset(const tensorDim_t dim, const Dimension& dims) {
  tensorSize_t res = 1; // minimum possible dimsize

  for(size_t idx = dims.nDims()-1; idx>dim; idx--){
    res *= dims.getItem(idx);
  }

  assert(res!=0);
  return res;
}

/**
 * @brief Like overload, but accepts negative dims.
 */
tensorSize_t Tensor::getDimOffset(const int dim, const Dimension& dims) {
  return getDimOffset(mapDim(dim, dims), dims);
}

/**
 * @brief No explanation needed.
 */
ftype Tensor::getItem(const std::vector<tensorDim_t>& idx) const {
  return (*values)[computeLinearIdx(idx, dims)]; 
}

/**
 * @brief Special getter, indexes the contained underlying array linearly.
 * Can lead to unexpected results in multidimensional tensors.
 */
ftype Tensor::getItem(tensorSize_t idx) const {
  return (*this)[idx];
}

/**
 * @brief For convenience.
 */
ftype Tensor::operator[](tensorSize_t idx) const {
  return (*values)[idx];
}


ftype Tensor::getItem(tensorDim_t idx0, tensorDim_t idx1) const {
  return getItem({idx0, idx1});
}

ftype Tensor::getItem(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) const {
  return getItem({idx0, idx1, idx2});
}

ftype Tensor::getItem(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) const {
  return getItem({idx0, idx1, idx2, idx3});
}

/**
 * @brief No explanation needed.
 */
void Tensor::setItem(ftype item, const std::vector<tensorDim_t>& idx) {
  (*values)[computeLinearIdx(idx, dims)] = item;
}

/**
 * @brief Special setter, indexes the contained underlying array linearly.
 * Can lead to unexpected results in multidimensional tensors.
 */
void Tensor::setItem(ftype item, tensorDim_t idx) { 
  (*values)[idx] = item;
}

void Tensor::setItem(ftype item, tensorDim_t idx0, tensorDim_t idx1) { 
  setItem(item, {idx0, idx1});
}

void Tensor::setItem(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) { 
  setItem(item, {idx0, idx1, idx2});
}

void Tensor::setItem(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) { 
  setItem(item, {idx0, idx1, idx2, idx3});
}