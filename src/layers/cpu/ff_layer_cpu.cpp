/**
 * @file ff_layer_cpu.cpp
 * @author Robert Baumgartner (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2025-10-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ff_layer_cpu.h"

#include <cstdlib>
#include <utility>

using namespace std;

ff_layer_cpu::ff_layer_cpu(const int in_size, const int out_size) : layer_base() {
    auto initializer = initializer_factory::get_initializer();
    
    weights.resize(out_size);
    for(int i=0; i<out_size; ++i){
        auto v = vector<ftype>(in_size);
        for(int j=0; j<in_size; ++j){
            v[j] = initializer->get_random_number();
        }
        weights[i] = move(v);
    }

    v1 = static_cast<ftype*>(malloc(out_size * sizeof(ftype)));
}

ff_layer_cpu::~ff_layer_cpu() noexcept {
    free(v1);
}

void ff_layer_cpu::reset_vector(ftype* v, const int size) const noexcept {
    for(int i=0; i<size; ++i){
        v[i]=0;
    }
}

ftype* ff_layer_cpu::forward(ftype* input) const {
    static const int in_size = weights.size();
    static const int out_size = weights[0].size();

    reset_vector(v1, out_size);
    
    for(int i=0; i<out_size; ++i){
        const auto& w = weights[i];
        for(int j=0; j<in_size; ++j){
            v1[i]+= input[i] * w[i];
        }
    }

    return v1;
}

ftype* ff_layer_cpu::backward(ftype* input) {

}