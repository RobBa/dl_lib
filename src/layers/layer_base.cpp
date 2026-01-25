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

using namespace layers;

ftype LayerBase::get() const {
  assert(weights);
  return weights.value().get();
}
ftype LayerBase::get(int idx) const {
  assert(weights);
  return weights.value().get(idx);
}
ftype LayerBase::get(int idx1, int idx2) const {
  assert(weights);
  return weights.value().get(idx1, idx2);
}
ftype LayerBase::get(int idx1, int idx2, int idx3) const {
  assert(weights);
  return weights.value().get(idx1, idx2, idx3);
}
ftype LayerBase::get(int idx1, int idx2, int idx3, int idx4) const {
  assert(weights);
  return weights.value().get(idx1, idx2, idx3, idx4);
}


void LayerBase::set(ftype item) {
  assert(weights);
  weights.value().set(item);
}
void LayerBase::set(ftype item, int idx) {
  assert(weights);
  weights.value().set(item, idx);
}
void LayerBase::set(ftype item, int idx1, int idx2) {
  assert(weights);
  weights.value().set(item, idx1, idx2);
}
void LayerBase::set(ftype item, int idx1, int idx2, int idx3) {
  assert(weights);
  weights.value().set(item, idx1, idx2, idx3);
}
void LayerBase::set(ftype item, int idx1, int idx2, int idx3, int idx4) {
  assert(weights);
  weights.value().set(item, idx1, idx2, idx3, idx4);
}