/**
 * @file sgd.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-03-08
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "sgd.h"

#ifdef __CUDA
#include "training/optimizers/cuda/optimizers.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace train;

void SgdOptimizer::step() {
  for(auto& t: params){
    auto grads = t->getGrads();
    switch(t->getDevice()) {
      case Device::CPU:
        for(auto idx=0; idx<t->getSize(); idx++){
          auto updatedWeight = (*t)[idx] - lr*(*grads)[idx];
          t->set(updatedWeight, idx);
        }
        break;
      case Device::CUDA:
      #ifdef __CUDA
        cuda_impl::sgdStep(*t, *grads, lr);
      #else
        __throw_invalid_argument("Attempted to use CUDA tensor");
      #endif
        break;
    }
  }
}
