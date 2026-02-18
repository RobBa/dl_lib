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

ftype LayerBase::getItem(vector<tensorDim_t>&&idx) const {
  assert(weights);
  return weights.value().getItem(std::move(idx));
}

void LayerBase::setItem(ftype item, vector<tensorDim_t>&& idx) {
  assert(weights);
  weights.value().setItem(item, std::move(idx));
}

void LayerBase::print(ostream& os) const noexcept {
  assert(weights);
  os << weights.value();
}

ostream& operator<<(ostream& os, const LayerBase& l) noexcept {
  l.print(os);
  return os;
}