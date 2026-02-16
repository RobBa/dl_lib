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

#include <cstdlib>
#include <utility>

using namespace std;
using namespace layers;

FfLayer::FfLayer(const tensorDim_t in_size, const tensorDim_t out_size) {
    //weights.emplace(Device::CPU, in_size, out_size);
    //weights->reset(utility::InitClass::Gaussian);
}

Tensor FfLayer::forward(const Tensor& input) const {
    return *weights * input;
}

//ftype* FfLayer::backward(ftype* input) {

//}