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

FfLayer::FfLayer(const std::uint16_t in_size, const std::uint16_t out_size){
    auto initializer = utility::InitializerFactory::getInitializer();

    weights = Tensor(in_size, out_size);
    v1 = static_cast<ftype*>(malloc(out_size * sizeof(ftype)));
}

FfLayer::~FfLayer() noexcept {
    free(v1);
}

void FfLayer::resetVector(ftype* v, const std::uint16_t size) const noexcept {
    for(int i=0; i<size; ++i){
        v[i]=0;
    }
}

ftype* FfLayer::forward(ftype* input) const {
    /* static const int in_size = weights.size();
    static const int out_size = weights[0].size();

    resetVector(v1, out_size);
    
    for(int i=0; i<out_size; ++i){
        const auto& w = weights[i];
        for(int j=0; j<in_size; ++j){
            v1[i]+= input[i] * w[i];
        }
    }

    return v1; */
}

//ftype* FfLayer::backward(ftype* input) {

//}