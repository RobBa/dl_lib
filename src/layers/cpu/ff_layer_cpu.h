#pragma once

#include "layer_base.h"
#include "initializers.h"

#include <vector>
#include <cstdlib>

template<typename T> // requires std::is_floating_point_v< std::remove_cv_t<T> > checked by base class already
class ff_layer_cpu : public layer_base<T> {
    protected:
        std::vector< std::vector<T> > weights;

    public:
        ff_layer_cpu(int in_size, int out_size);
        
        T* forward(T* input) const override;
        T* backward(T* input) override;
};

template<typename T>
T* ff_layer_cpu<T>::forward(T* input) const {
    static const int in_size = weights.size();
    static const int out_size = weights[0].size();
    
    static const std::size_t res_in_bytes = out_size * sizeof(T); 
    T* res = static_cast<T*>(std::malloc(res_in_bytes));

    

    std::free(input);
}

template<typename T>
T* ff_layer_cpu<T>::backward(T* input) {

}

template<typename T>
ff_layer_cpu<T>::ff_layer_cpu(const int in_size, const int out_size) : layer_base<T>() {
    auto initializer = initializer_factory::get_initializer();
    
    weights.resize(in_size);
    for(int i=0; i<in_size; ++i){
        auto v = std::vector<T>(out_size);
        for(int j=0; j<out_size; ++j){
            if constexpr(std::is_same_v<T, double>){
                v[j] = initializer->get_random_number();
            }
            else if constexpr(std::is_same_v<T, float>){
                v[j] = static_cast<float>(initializer->get_random_number());
            }
        }
        weights[i] = move(v);
    }
}