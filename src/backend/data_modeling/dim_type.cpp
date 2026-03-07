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

tensorDim_t Dimension::multVector(const std::vector<tensorDim_t>& dims) const noexcept {
  tensorDim_t res = 1;

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

  if(size==0){
    __throw_invalid_argument("Tensor-Dims must all be greater than 0.");
  }
}

/**
 * @brief Swap along the two given dimensions.
 */
void Dimension::swap(const tensorDim_t dim1, const tensorDim_t dim2) {
  auto tmp = dims[dim1];
  dims[dim1] = dims[dim2];
  dims[dim2] = tmp;
}

Dimension::Dimension(const vector<tensorDim_t>& dims) : dims{dims} {
  size = multVector(dims);

  if(size==0){
    __throw_invalid_argument("Tensor-Dims must all be greater than 0.");
  }
}

Dimension::Dimension(const Dimension& other) : dims{other.dims}, size{other.size} { }

Dimension& Dimension::operator=(const Dimension& other) {
  if(this==&other) return *this;

  dims = other.dims;
  size = other.size;

  return *this;
}

Dimension::Dimension(Dimension&& other) noexcept : dims{move(other.dims)}, size{other.size} {}

Dimension& Dimension::operator=(Dimension&& other) noexcept {
  if(this==&other) return *this;

  dims = move(other.dims);
  size = other.size;
  return *this;
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
  auto mappedIdx = getItem(idx);

  std::vector<tensorDim_t> newDims;
  newDims.reserve(dims.size() - 1);
  newDims.insert(newDims.end(), dims.begin(), dims.begin() + idx);
  newDims.insert(newDims.end(), dims.begin() + idx + 1, dims.end());

  return Dimension(newDims);
}

ostream& operator<<(ostream& os, const Dimension& d) noexcept {
  os << "(";
  for(int i=0; i<d.nDims(); i++){
    os << d.getItem(i);

    if(i+1<d.nDims()){
      os << ",";
    }
  }
  os << ")";
  return os;
}