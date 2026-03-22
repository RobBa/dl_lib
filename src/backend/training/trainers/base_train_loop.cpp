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
#include <random>

#include <iostream>

using namespace std;
using namespace train;

void BaseTrainLoop::run(shared_ptr<Tensor>& x, shared_ptr<Tensor>& y, const bool shuffle, const bool verbose) {
  const auto nSamples = x->getDims()[0];

  for(size_t e=0; e<epochs; e++){
    std::vector<tensorDim_t> indices(nSamples);
    std::iota(indices.begin(), indices.end(), 0);

    if(verbose)
      cout << "\nEpoch " << e;

    if(shuffle){
      std::random_device rd;
      std::mt19937 rng(rd());
      std::shuffle(indices.begin(), indices.end(), rng);
    }

    tensorDim_t low = 0;

    int batch = 0;
    while(low < nSamples){
      if(verbose)
        cout << "\nBatch " << batch << endl;

      std::span<const tensorDim_t> batchSpan(indices.data() + low, low+bsize < nSamples ? bsize : nSamples-low);

      auto xBatch = make_shared<Tensor>(x->getSlice(batchSpan));
      auto yBatch = make_shared<Tensor>(y->getSlice(batchSpan));

      auto yPred = (*graph)(xBatch);
      auto l = (*loss)(yBatch, yPred);

      l->backward();
      optim->step();
      optim->zeroGrad();

      low += bsize;
    }
  }
}