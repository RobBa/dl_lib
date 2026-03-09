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
#include "activation_functions/activation_function_base.h"
#include "data_modeling/tensor_functions.h"

#include "computational_graph/tensor_ops/graph_creation.h"

#include <cstdlib>
#include <utility>

using namespace std;
using namespace layers;

FfLayer::FfLayer(const vector<tensorDim_t>& dims, bool useBias, bool requiresGrad) 
    : FfLayer(dims, Tensor::getDefaultDevice(), useBias, requiresGrad) {}

/**
 * @brief Construct a new Ff Layer:: Ff Layer object
 * Assumption for dims: (in-size, out-size)
 * @param dims Dimensions, see above.
 * @param d The device.
 * @param useBias Use a bias if true. Bias will receiver shape (n_rows)
 * @param requiresGrad If true train this layer.
 */
FfLayer::FfLayer(const vector<tensorDim_t>& dims, Device d, bool useBias, bool requiresGrad)
  : LayerBase(useBias, requiresGrad) {
  assert(dims.size()==2);

  weights = make_shared<Tensor>(Dimension({dims[0], dims[1]}), d, requiresGrad);
  TensorFunctions::ToGaussian(*weights);
    
  if(useBias){
    bias = make_shared<Tensor>(vector<tensorDim_t>{dims[1]}, d, requiresGrad);
    TensorFunctions::ToGaussian(*bias);
  }
}

/**
 * @brief Normal forward function. Does not build computational graph.
 * 
 * Assumption for input: (b-size, ..., dim1, in-size)
 */
Tensor FfLayer::forward(const Tensor& input) const {
    auto res = input.matmul(*weights);

    if(useBias){
        res = res + *bias;
    }

    for(auto& af: activations){
      res = (*af)(res);
    }

    return res;
}

/**
 * @brief Like overload, but creates computational graph.
 */
std::shared_ptr<Tensor> FfLayer::forward(const std::shared_ptr<Tensor>& input) const {
    auto res = graph::matmul(input, weights);
    if(useBias){
        res = graph::add(res, bias); // TODO: add needs to happen on each of those, how to broadcast?
    }

    for(auto& af: activations){
      res = (*af)(res);
    }

    return res;  
}

void FfLayer::print(ostream& os) const noexcept {
    LayerBase::print(os);
    os << "\nuseBias: " << useBias ? "true" : "false";
}