#pragma once

#include "global_params.h"

#include <type_traits>

namespace layers {
    /** 
     * The base class for all the layers that we have. Not instantiable.
     */
    class layer_base {
        public:
            layer_base() = default;
            virtual ~layer_base() noexcept {};

            virtual ftype* forward(ftype* input) const = 0;
            //virtual ftype* backward(ftype* input) = 0;
    };
}