/**
 * @file dim_type.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-18
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "dim_type.h"
#include "utility/safe_arithmetics.h"

#include <utility>
#include <sstream>

using namespace std;

tensorSize_t Dimension::multVector(const std::vector<tensorDim_t>& dims) const noexcept {
  tensorSize_t res = 1;

#ifndef NDEBUG
  utility::SafeArithmetics_t<tensorSize_t> mult(1);
  for(auto dim: dims){
    mult = mult * dim;
  }

  res = mult.value;
#else 
  for(auto dim: dims){
    res *= dim;
  }
#endif // NDEBUG

  return res;
}

void Dimension::resize(const std::vector<tensorDim_t>& dims) {
  this->dims = dims;
  size = multVector(dims);
  assert(size>0);
}

/**
 * @brief Swap along the two given dimensions.
 */
void Dimension::swap(const tensorDim_t dim1, const tensorDim_t dim2) {
  auto swapArr = [dim1, dim2](array<tensorSize_t, MAX_NDIMS>& arr){
    auto tmp = arr[dim1];
    arr[dim1] = arr[dim2];
    arr[dim2] = tmp;
  };
  
  auto tmp = dims[dim1];
  dims[dim1] = dims[dim2];
  dims[dim2] = tmp;

  swapArr(strides);
}

Dimension::Dimension(const vector<tensorDim_t>& dims) : dims{dims}, strides{} {
  size = multVector(dims);
  lastDimIdx = dims.size()-1;
  resetStrides();
  assert(size>0);
}

Dimension::Dimension(vector<tensorDim_t>&& dims, array<tensorSize_t, MAX_NDIMS>&& strides)
  : dims{dims}, strides{strides} 
{
  size = multVector(dims);
  lastDimIdx = dims.size()-1;
}

Dimension::Dimension(const Dimension& other) 
  : dims{other.dims}, strides{other.strides}, size{other.size}, lastDimIdx{other.lastDimIdx}
{}

Dimension& Dimension::operator=(const Dimension& other) {
  if(this==&other) return *this;

  dims = other.dims;
  strides = other.strides;
  size = other.size;
  lastDimIdx = other.lastDimIdx;

  return *this;
}

Dimension::Dimension(Dimension&& other) noexcept 
  : dims{move(other.dims)}, strides{std::move(other.strides)}, size{other.size}, lastDimIdx{other.lastDimIdx} 
{}

Dimension& Dimension::operator=(Dimension&& other) noexcept {
  if(this==&other) return *this;

  dims = move(other.dims);
  strides = std::move(other.strides);
  size = other.size;
  lastDimIdx = other.lastDimIdx;

  return *this;
}

/**
 * @brief Computes the strides as they are;
 */
void Dimension::resetStrides() noexcept {
  tensorSize_t res=1;
  strides[lastDimIdx] = res;
  for(tensorDim_t i=lastDimIdx-1; i>=0; i++){
    strides[i] = strides[i+1] * dims[i+1];
  }
}

tensorSize_t Dimension::getStride(const int i) const noexcept {
  if(i<0)
    return strides[lastDimIdx + i + 1];
  return strides[i];
}

/**
 * @brief This method gets interesting when we want to get a copy of 
 * this dimension instance, but we collapsed one of the dimensions.
 * E.g. when we have a tensor, and we sum over one of its dimensions 
 * to get a new tensor, then this will be the new dimensions of the result.
 * 
 * Example: t=Tensor with dims (b-size, d). We sum over all batches and 
 * get a new tensor tSum=Tensor with dims (d).
 * 
 * @param idx The dimension to collapse.
 */
Dimension Dimension::collapseDimension(int idx) const {
  auto mappedIdx = get(idx);

  std::vector<tensorDim_t> newDims;
  newDims.reserve(dims.size() - 1);
  newDims.insert(newDims.end(), dims.begin(), dims.begin() + idx);
  newDims.insert(newDims.end(), dims.begin() + idx + 1, dims.end());

  std::array<tensorSize_t, MAX_NDIMS> newStrides{};
  tensorDim_t strideIdx = 0;
  for(tensorDim_t i=0; i<strides.size(); i++){
    if(i==mappedIdx)
      continue;

    newStrides[strideIdx] = strides[i];
  }

  return Dimension(std::move(newDims), std::move(newStrides));
}

ostream& operator<<(ostream& os, const Dimension& d) noexcept {
  if(d.size>0){
    os << "\n(";
    for(int i=0; i<d.nDims(); i++){
      os << d.get(i);

      if(i+1<d.nDims()){
        os << ",";
      }
    }
    os << ")";
    return os;
  }

  os << "\nempty";
  return os;
}