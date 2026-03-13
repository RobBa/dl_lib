/**
 * @file sequential.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "sequential.h"

using namespace std;
using namespace module;

Tensor Sequential::operator()(const Tensor& input) const {
  if(layers.size()==0){
    __throw_invalid_argument("Network empy, cannot be called.");
  }

  auto x = layers[0]->operator()(input);
  for(int i=1; i<layers.size(); i++){
    x = layers[i]->operator()(x);
  }

  return x;
}

shared_ptr<Tensor> Sequential::operator()(const shared_ptr<Tensor>& input) const {
  if(layers.size()==0){
    __throw_invalid_argument("Network empy, cannot be called.");
  }

  auto x = layers[0]->operator()(input);
  for(int i=1; i<layers.size(); i++){
    x = layers[i]->operator()(x);
  }

  return x;
}

vector<shared_ptr<Tensor>> Sequential::parameters() const {
  vector<shared_ptr<Tensor>> res;

  for(const auto& layer: layers) {
    auto p = layer->parameters();
    for(auto& pp: p){
      if(pp){
        res.push_back(std::move(pp));
      }
    }
  }

  return res;
}

void Sequential::append(shared_ptr<module::ModuleBase> l) {
  layers.push_back(move(l));
}