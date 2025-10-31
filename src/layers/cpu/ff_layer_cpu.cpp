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

#include "initializers.h"

#include <algorithm>
#include <utility>

using namespace std;

template<typename T> 
ff_layer_cpu<T>::ff_layer_cpu(size_t in_size, size_t out_size) : layer_base<T>() {
    auto initializer = initializer_factory::get_initializer();
    
    weights.resize(in_size);
    for(int i=0; i<in_size; ++i){
        auto v = std::vector<T>(out_size);
        for(int j=0; j<out_size; ++j){
            if constexpr(is_same_v<T, double>){
                v[j] = initializer->get_random_number();
            }
            else if constexpr(is_same_v<T, float>){
                v[j] = static_cast<float>(initializer->get_random_number());
            }
            else{
                static_assert(false, "T must be of floating point type");
            }
        }
        weights[i] = move(v); // TODO: use emplace function instead?
    }
}

