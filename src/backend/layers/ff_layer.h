/**
 * @file ff_layer.h
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
#include "utility/initializers.h"

#include <optional>

namespace layers {
    class FfLayer : public LayerBase {
        protected:            
            // memoization 
            // TODO: necessary?
            //mutable std::optional<Tensor> v1;

        public:
            FfLayer(tensorDim_t in_size, tensorDim_t out_size);

            Tensor forward(const Tensor& input) const override;
            //ftype* backward(ftype* input) override;
    };
}
