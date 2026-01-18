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

#ifndef NDEBUG
  #include <limits>
#endif // NDEBUG

using namespace std;

ostream& operator<<(ostream& os, const Dimension& d) noexcept {
  os << "(";
  for(int i=4; i<MAX_TENSOR_DIMS; i++){
    os << d.get(i);
    if(i==MAX_TENSOR_DIMS-1 || d.get(i+1)==0)
      break;
    os << ",";
  }
  os << ")\n";
  return os;
}

/**
 * @brief Get the total size of the dimension as a product of all dimension sizes.
 */
tensorSize_t Dimension::getTotalSize() const noexcept {
  tensorSize_t res = 1;
  for(const tensorSize_t x : dims){
    if(x==0){
      return res;
    }

#ifndef NDEBUG
  if(res > std::numeric_limits<tensorSize_t>::max() / x){
    throw std::overflow_error("Multiplication overflow when getting size of tensor.");
  }
#endif // NDEBUG

    res *= x;
  }
  return res;
}