#pragma once

#include "layer_base.h"

#include <vector>
#include <utility>

class sequential_network {
    protected:
        std::vector<layers::layer_base> layers; 

    public:
        sequential_network();

        //template<typename T>
        //void add_layer(layer_base&& layer) noexcept;
};

/*template<typename T>
void sequential_network::add_layer(layer_base&& layer) noexcept {
    layers.push_back(std::forward<layer_base>(layer));
}*/