/**
 * @file layer_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "global_params.h"
#include "tensor.h"

#include <optional>

namespace layers {
    /** 
     * The base class for all the layers that we have. Not instantiable.
     */
    class LayerBase {       
        protected:
            std::optional<Tensor> weights = std::nullopt;

        public:
            LayerBase() = default;
            virtual ~LayerBase() noexcept = default;

            virtual Tensor forward(const Tensor& input) const = 0;
            //virtual ftype* backward(ftype* input) = 0;

            // weights should always exist, never nullopt outside of c'tor
            const Dimension& getDims() const noexcept { 
                return weights.value().getDims(); 
            }
    };
}