/**
 * @file optimizer_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "optimizer_base.h"

#include <span>

#include <numeric>
#include <algorithm>

using namespace std;
using namespace train;

void OptimizerBase::run(shared_ptr<Tensor>& x, shared_ptr<Tensor>& y, const bool shuffle) {
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
      step(make_shared<Tensor>(x->getSlice(batchSpan)), make_shared<Tensor>(y->getSlice(batchSpan)));
      low += bsize;
    }
  }
}