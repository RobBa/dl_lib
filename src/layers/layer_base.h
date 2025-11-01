#pragma once

#include <type_traits>

/** 
 * The base class for all the layers that we have. Not instantiable.
 */
template<typename T> requires std::is_floating_point_v< std::remove_cv_t<T> >
class layer_base {
    protected:
        layer_base(){};
    
    public:
        virtual T* forward(T* input) const;
        virtual T* backward(T* input);
};