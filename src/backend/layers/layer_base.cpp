/**
 * @file layer_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "layer_base.h"

#include <utility>

using namespace std;
using namespace layers;

void LayerBase::print(ostream& os) const noexcept {
  assert(weights);
  
  os << "Weigths:\n";
  os << *weights;

  if(bias){
    os << "Bias:\n";
    os << *bias;
  }
}

ostream& operator<<(ostream& os, const LayerBase& l) noexcept {
  l.print(os);
  return os;
}