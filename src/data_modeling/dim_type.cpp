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
#include "safe_arithmetics.h"

#include <utility>

using namespace std;

tensorDim_t Dimension::multVector(const std::vector<tensorDim_t>& dims) const noexcept {
  tensorDim_t res = 1;

#ifndef NDEBUG
  SafeArithmetics_t<tensorSize_t> mult(1);
  for(auto dim: dims){
    mult * dim;
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
  tensorDim_t tmp = dims[dim1];
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

ostream& operator<<(ostream& os, const Dimension& d) noexcept {
  os << "(";
  for(int i=0; i<d.nDims(); i++){
    os << d.get(i);
    os << ",";
  }
  os << ")\n";
  return os;
}