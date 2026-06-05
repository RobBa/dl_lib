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

#include <cmath>
#include <stdexcept>

#ifdef __CUDA
#include "training/optimizers/cuda/optimizers.cuh"
#endif

using namespace std;
using namespace train;

void OptimizerBase::zeroGrad() noexcept
{
  for (auto& p: params){
    auto grads = p->getGrads();

    if(grads)
      TensorFunctions::ToZeros(*grads);
  }
}

void OptimizerBase::clipGradients(ftype maxNorm) noexcept
{
  switch(params[0]->getDevice()) {
    case Device::CPU:
    {
      // compute global L2 norm across all parameters
      ftype totalNorm = 0.0f;
      for (const auto &param : params)
      {
        auto grads = param->getGrads();
        if (!grads)
          continue;
        for (tensorSize_t i = 0; i < grads->getSize(); i++)
        {
          auto g = (*grads)[i];
          totalNorm += g * g;
        }
      }
      totalNorm = std::sqrt(totalNorm);

      if (totalNorm > maxNorm)
      {
        const ftype scale = maxNorm / (totalNorm + EPS_OPTIM_GRADCLIP);
        for (const auto &param : params)
        {
          auto grads = param->getGrads();
          if (!grads)
            continue;
          for (tensorSize_t i = 0; i < grads->getSize(); i++)
          {
            grads->set((*grads)[i] * scale, i);
          }
        }
      }
    }
    break;
    case Device::CUDA:
    {
    #ifdef __CUDA
      cuda_impl::clipGradients(params, maxNorm);
    #else
      __throw_invalid_argument("Not compiled with CUDA.");
    #endif
    }
    break;
  }
}