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
using namespace layers;

/**
 * @brief Returns true if dimensions valid, else false. 
 * Ensures consistency along network.
 */
bool SequentialNetwork::assertDims(const LayerBase& layer) const noexcept {
  if(layers.size() == 0)
    return true;

  return layers[layers.size()-1]->getDims() == layer.getDims(); 
}

Tensor SequentialNetwork::forward(const Tensor& input) const {
  if(input.getDims().getItem(-1) != layers[0]->getDims().getItem(-2)){
    __throw_invalid_argument("Input tensor has invalid dimension.");
  }

  if(layers.size()==0){
    __throw_invalid_argument("Network empy, cannot be called.");
  }

  auto x = layers[0]->forward(input);
  for(int i=1; i<layers.size(); i++){
    x = layers[i]->forward(x);
  }

  return x;
}

std::shared_ptr<Tensor> SequentialNetwork::forward(const std::shared_ptr<Tensor>& input) const {
  if(input->getDims().getItem(-1) != layers[0]->getDims().getItem(-2)){
    __throw_invalid_argument("Input tensor has invalid dimension.");
  }

  if(layers.size()==0){
    __throw_invalid_argument("Network empy, cannot be called.");
  }

  auto x = layers[0]->forward(input);
  for(int i=1; i<layers.size(); i++){
    x = layers[i]->forward(x);
  }

  return x;
}

std::vector<std::shared_ptr<Tensor>> SequentialNetwork::getParams() const {
  std::vector<std::shared_ptr<Tensor>> res;
  res.reserve(layers.size()*2);

  for(const auto& layer: layers){
    auto [weigths, bias] = layer->getParams();
    res.push_back(std::move(weights));
    res.push_back(std::move(bias));
  }
  
  return res;
}

void SequentialNetwork::append(shared_ptr<layers::LayerBase> l) {
  if(!assertDims(*l)){
    __throw_invalid_argument("Dimensions of tensors don't fit.");
  }
  layers.push_back(std::move(l));
}

void SequentialNetwork::append(shared_ptr<activation::ActivationFunctionBase> f) {
  assert(layers.size()>0);
  layers[layers.size()-1]->addActivation(std::move(f));
}