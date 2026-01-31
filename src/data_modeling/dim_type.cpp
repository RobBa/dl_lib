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

using namespace std;

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