/**
 * @file sequential.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "layer_base.h"

#include <vector>
#include <utility>

class SequentialNetwork {
    protected:
        std::vector<layers::LayerBase> layers; 

    public:
        SequentialNetwork();

        //template<typename T>
        //void addLayer(LayerBase&& layer) noexcept;
};

/*template<typename T>
void SequentialNetwork::addLayer(LayerBase&& layer) noexcept {
    layers.push_back(std::forward<LayerBase>(layer));
}*/