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
#include <type_traits>

class SequentialNetwork {
    protected:
        std::vector<layers::LayerBase> layers;
        bool assertDims(const layers::LayerBase& layer) const noexcept;

    public:
        SequentialNetwork() = default;

        template <typename T> requires (std::derived_from<T, layers::LayerBase>)
        void addLayer(T&& layer) {
            if(!assertDims(layer)){
                // TODO: show warning that the dims don't match
                return;
            }
            layers.push_back(std::forward<T>(layer));
        }

        //template<typename T>
        //void addLayer(LayerBase&& layer) noexcept;
};

/*template<typename T>
void SequentialNetwork::addLayer(LayerBase&& layer) noexcept {
    layers.push_back(std::forward<LayerBase>(layer));
}*/