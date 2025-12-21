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
#include "dim_type.h"

#include <array>
#include <type_traits>
#include <tuple>
#include <cstdint>

namespace layers {
    /** 
     * The base class for all the layers that we have. Not instantiable.
     */
    class LayerBase {            
        public:
            LayerBase() = default;
            virtual ~LayerBase() noexcept {};

            virtual ftype* forward(ftype* input) const = 0;
            //virtual ftype* backward(ftype* input) = 0;
            virtual const Dimension& getDim() const noexcept = 0;
    };
}