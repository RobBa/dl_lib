#pragma once

#include "layer_base.h"

#include <vector>

template<typename T>
class ff_layer_cpu : public layer_base<T> {
    protected:
        std::vector< std::vector<T> > weights;

    public:
        ff_layer_cpu(size_t in_size, size_t out_size);
        
        T* forward(T* input) const override;
        T* backward(T* input) override;
};

template<typename T>
T* ff_layer_cpu<T>::forward(T* input) const {
    
}

template<typename T>
T* ff_layer_cpu<T>::backward(T* input) {

}