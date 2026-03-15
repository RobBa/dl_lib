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

using namespace std;
using namespace module;

FfLayer::FfLayer(tensorDim_t inSize, tensorDim_t outSize, bool useBias, bool requiresGrad) 
    : FfLayer(inSize, outSize, Tensor::getDefaultDevice(), useBias, requiresGrad) {}

/**
 * @brief Construct a new Ff Layer:: Ff Layer object
 * Assumption for dims: (in-size, out-size)
 * @param dims Dimensions, see above.
 * @param d The device.
 * @param useBias Use a bias if true. Bias will receiver shape (n_rows)
 * @param requiresGrad If true train this layer.
 */
FfLayer::FfLayer(tensorDim_t inSize, tensorDim_t outSize, Device d, bool useBias, bool requiresGrad)
  : useBias{useBias}, requiresGrad{requiresGrad} 
{
  weights = make_shared<Tensor>(Dimension({inSize, outSize}), d, requiresGrad);
  TensorFunctions::ToGaussian(*weights, 0, 0.1);
  weights = weights;
    
  if(useBias){
    bias = make_shared<Tensor>(vector<tensorDim_t>{outSize}, d, requiresGrad);
    TensorFunctions::ToGaussian(*bias, 0, 0.001);
    bias = bias;
  }
}

/**
 * @brief Normal forward function. Does not build computational graph.
 * 
 * Assumption for input: (b-size, ..., dim1, in-size)
 */
Tensor FfLayer::operator()(const Tensor& input) const {
  auto res = input.matmul(*weights);

  if(useBias){
    res = res + *bias;
  }

  return res;
}

/**
 * @brief Like overload, but creates computational graph.
 */
std::shared_ptr<Tensor> FfLayer::operator()(const std::shared_ptr<Tensor>& input) const {
  auto res = cgraph::matmul(input, weights);
  if(useBias){
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