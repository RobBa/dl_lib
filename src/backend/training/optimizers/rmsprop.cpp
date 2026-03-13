/**
 * @file rmsprop.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "rmsprop.h"

using namespace std;
using namespace train;

void RmsPropOptimizer::step() {
  constexpr ftype eps = 1e-6;
  for(const auto& param: params){
    auto tPtr = param.get();
    const auto gPtr = tPtr->getGrads().get();
    auto vPtr = movingAvg[tPtr].get();

    // update moving avg
    if(vPtr!=nullptr) { // hot path
      for(tensorSize_t i=0; i<gPtr->getSize(); i++){ 
        auto g = (*gPtr)[i];
        auto update = decay * (*vPtr)[i] + (1-decay)*g*g;
        vPtr->setItem(update, i);
      }
    }
    else { // init loop
      movingAvg[tPtr] = make_unique<Tensor>(tPtr->getDims(), tPtr->getDevice(), false); // create empty tensor
      vPtr = movingAvg[tPtr].get();
      for(tensorSize_t i=0; i<tPtr->getSize(); i++) {
        auto g = (*tPtr)[i];
        vPtr->setItem((1-decay)*g*g, i);
      }
    }

    // update gradients
    for(tensorSize_t i=0; i<tPtr->getSize(); i++) {
      auto update = (*tPtr)[i] - lr * (*gPtr)[i] / ((*vPtr)[i] + eps);
      tPtr->setItem(update, i);
    }
  }
}