/**
 * @file ff_layer.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ff_layer.h"
#include "data_modeling/tensor_functions.h"

#include "computational_graph/tensor_ops/graph_creation.h"

#include <cstdlib>
#include <utility>

#ifdef __CUDA
#include "module/layers/cuda/layers.cuh"
#else
#include <stdexcept>
#endif

using namespace std;
using namespace module;
using namespace utility;

FfLayer::FfLayer(tensorDim_t inSize, tensorDim_t outSize, bool useBias, bool requiresGrad, shared_ptr<InitializerBase> init) 
    : FfLayer(inSize, outSize, Tensor::getDefaultDevice(), useBias, requiresGrad, init) {}

/**
 * @brief Construct a new Ff Layer:: Ff Layer object
 * Assumption for dims: (in-size, out-size)
 * @param dims Dimensions, see above.
 * @param d The device.
 * @param useBias Use a bias if true. Bias will receiver shape (n_rows)
 * @param requiresGrad If true train this layer.
 */
FfLayer::FfLayer(tensorDim_t inSize, tensorDim_t outSize, Device d, 
    bool useBias, bool requiresGrad, shared_ptr<InitializerBase> init)
  : requiresGrad{requiresGrad} 
{
  if(!init){
    init = make_shared<NormalXavierInitializer>(inSize, outSize);  
  }

  weights = make_shared<Tensor>(Dimension({inSize, outSize}), d, requiresGrad);
  weights->reset(init);
    
  if(useBias){
    bias = make_shared<Tensor>(vector<tensorDim_t>{outSize}, d, requiresGrad);
    TensorFunctions::ToZeros(*bias);
  }
}

/**
 * @brief Normal forward function. Does not build computational graph.
 * 
 * Assumption for input: (b-size, ..., dim1, in-size)
 */
Tensor FfLayer::operator()(const Tensor& input) const {
  switch(input.getDevice()) {
    case Device::CPU: {
      auto res = input.matmul(*weights);
      if(bias) res = res + *bias;
      return res;
    }
    case Device::CUDA:
    #ifdef __CUDA
      {
        if(bias) {
          // TODO: the following should be an optimized fusion kernel
          /* auto resDims = input.getDims().toVector();
          resDims.back() = static_cast<tensorDim_t>(weights->getDims().get(-1));
          Tensor res(resDims, input.getDevice(), false);

          cuda_impl::matMulPlusBias(res, input, *weights, *bias); */

          auto res = input.matmul(*weights);
          res = res + *bias;
          return res;
        } else {
          return input.matmul(*weights);
        }
      }
    #else
      __throw_invalid_argument("Attempted to give CUDA tensor");
      return input.createShallowCopy(); // line should not be reached
    #endif
  }
}

/**
 * @brief Like overload, but creates computational graph.
 */
std::shared_ptr<Tensor> FfLayer::operator()(const std::shared_ptr<Tensor>& input) const {
  // TODO: if you fuse kernel you'll also have to do it here. Perhaps give tensor a matmulplusbias method?
  auto res = cgraph::matmul(input, weights);
  if(bias){
    res = cgraph::add(res, bias);
  }

  return res;  
}

void FfLayer::print(ostream& os) const noexcept {
  os << "\nFfLayer\nWeigths:\n" << *weights;
  if(bias){
    os << "\nBias:\n" << *bias;
  }
}