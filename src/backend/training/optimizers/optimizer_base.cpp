/**
 * @file optimizer_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "optimizer_base.h"

#include "data_modeling/tensor_functions.h"

using namespace train;

void OptimizerBase::zeroGrad() noexcept{ 
  for(auto& p: params){
    auto grads = p->getGrads();
    
    if(grads)
      TensorFunctions::ToZeros(*grads);
  }
}