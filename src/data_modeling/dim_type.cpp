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

#include <utility>

using namespace std;

Dimension::Dimension(const Dimension& other) : dims{other.dims} { }

Dimension& Dimension::operator=(const Dimension& other) {
  if(this==&other) return *this;

  dims = other.dims;
  return *this;
}

Dimension::Dimension(Dimension&& other) noexcept : dims{move(other.dims)} { }

Dimension& Dimension::operator=(Dimension&& other) noexcept {
  if(this==&other) return *this;

  dims = move(other.dims);
  return *this;
}

ostream& operator<<(ostream& os, const Dimension& d) noexcept {
  os << "(";
  for(int i=0; i<MAX_TENSOR_DIMS; i++){
    os << d.get(i);
    if(i==MAX_TENSOR_DIMS-1 || d.get(i+1)==0)
      break;
    os << ",";
  }
  os << ")\n";
  return os;
}