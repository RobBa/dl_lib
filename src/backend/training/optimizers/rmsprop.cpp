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

#ifdef __CUDA
#include "training/optimizers/cuda/optimizers.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace train;

void RmsPropOptimizer::step() {
  constexpr ftype eps = 1e-8;
  for(const auto& param: params){
    auto tPtr = param.get();
    const auto gPtr = tPtr->getGrads().get();

    switch(tPtr->getDevice()) {
      case Device::CPU: {
        auto vPtr = movingAvg[tPtr].get();

        if(vPtr != nullptr) {
          for(tensorSize_t i=0; i<gPtr->getSize(); i++){
            auto g = (*gPtr)[i];
            auto update = decay * (*vPtr)[i] + (1-decay)*g*g;
            vPtr->set(update, i);
          }
        }
        else {
          movingAvg[tPtr] = make_unique<Tensor>(tPtr->getDims(), tPtr->getDevice(), false);
          vPtr = movingAvg[tPtr].get();
          for(tensorSize_t i=0; i<tPtr->getSize(); i++) {
            auto g = (*gPtr)[i];
            vPtr->set((1-decay)*g*g, i);
          }
        }

        for(tensorSize_t i=0; i<tPtr->getSize(); i++) {
          auto update = (*tPtr)[i] - lr * (*gPtr)[i] / ((*vPtr)[i] + eps);
          tPtr->set(update, i);
        }
        break;
      }
      case Device::CUDA:
      #ifdef __CUDA
        if(movingAvg[tPtr] == nullptr) {
          movingAvg[tPtr] = make_unique<Tensor>(tPtr->getDims(), tPtr->getDevice(), false);
        }
        cuda_impl::rmspropStep(*tPtr, *movingAvg[tPtr], *gPtr, lr, decay, eps);
      #else
        __throw_invalid_argument("Attempted to use CUDA tensor");
      #endif
        break;
    }
  }
}
