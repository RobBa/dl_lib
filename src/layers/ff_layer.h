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
#include "initializers.h"

#include <vector>
#include <memory>

namespace layers {
    class FfLayer : public LayerBase {
        protected:            
            // memoization
            mutable Tensor v1;

            void resetVector(ftype* v, std::uint16_t size) const noexcept;

        public:
            FfLayer(std::uint16_t in_size, std::uint16_t out_size);

            Tensor forward(const Tensor& input) const override;
            //ftype* backward(ftype* input) override;
    };
}
