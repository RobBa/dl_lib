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

FfLayer::FfLayer(const std::uint16_t in_size, const std::uint16_t out_size) {
    auto initializer = utility::InitializerFactory::getInitializer();

    weights.emplace(in_size, out_size);
    // TODO: init the weigths randomly

    //v1 = make_unique<Tensor>(out_size);
}

/* void FfLayer::resetVector(ftype* v, const std::uint16_t size) const noexcept {
    for(int i=0; i<size; ++i){
        v[i]=0;
    }
} */

Tensor FfLayer::forward(const Tensor& input) const {
    return *weights * input;
}

//ftype* FfLayer::backward(ftype* input) {

//}