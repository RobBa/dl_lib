/**
 * @file base_train_loop.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "base_train_loop.h"

#include <span>

#include <numeric>
#include <algorithm>

using namespace std;
using namespace train;

void BaseTrainLoop::run(shared_ptr<Tensor>& x, shared_ptr<Tensor>& y, const bool shuffle) {
  for(size_t e=0; e<epochs; e++){
      std::vector<tensorDim_t> indices(bsize);
      std::iota(indices.begin(), indices.end(), 0);

    if(shuffle){
      std::random_shuffle(indices.begin(), indices.end());
    }

    const auto nSamples = x->getDims().getItem(0);
    tensorDim_t low = 0;
    while(low < nSamples){
      std::span<const tensorDim_t> batchSpan(indices.data() + low, low+bsize < nSamples ? bsize : nSamples-low);

      auto xBatch = make_shared<Tensor>(x->getSlice(batchSpan));
      auto yBatch = y->getSlice(batchSpan);

      auto yPred = network->forward(xBatch);
      auto l = (*loss)(yBatch, yPred);
      
      l->backward();
      optim->step();

      low += bsize;
    }
  }
}