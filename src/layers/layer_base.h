#pragma once

#include "global_params.h"

#include <type_traits>

/** 
 * The base class for all the layers that we have. Not instantiable.
 */
class layer_base {
    protected:
        layer_base(){};
    
    public:
        virtual ftype* forward(ftype* input) const = 0;
        virtual ftype* backward(ftype* input) = 0;

        virtual ~layer_base() noexcept {};
    };