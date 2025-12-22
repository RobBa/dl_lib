/**
 * @file tensor.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "dim_type.h"

#include <stdlib.h>
#include <unordered_map>

#include <concepts>
#include <cassert>

enum class Device {
    CPU,
    CUDA
};

consteval const char* DeviceToString(Device d) {
    switch(d){
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
    }
}

struct Tensor final {
    private:
        ftype* values = nullptr;
        Dimension dims;

        Device device;

        /**
         * @brief Folding expression since C++17: Does the 
         * product of the variadic templated types and returns 
         * them.
         */
        template<typename... T>
        auto varProduct(T... x){
            return (x * ...);
        }

        template<typename T>
        requires (std::is_integral_v< std::remove_const_t<T> >)
        void allocValues(const T size, const Device d) {
            switch(d){
                case Device::CPU:
                    values = static_cast<ftype*>( malloc(size * sizeof(ftype)) );
                case Device::CUDA:
                    std::__throw_invalid_argument("Not implemented yet.");
            }
        }

    public:
        Tensor() = default;
        ~Tensor() noexcept;
        
        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, Device d=Device::CPU)
            : device(d) {
            assert(dim1 >= 0);

            dims[0] = dim1;
            allocValues(dim1, d);
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, Device d=Device::CPU)
            : device(d) {
            assert(dim1 >= 0);
            assert(dim2 >= 0);

            dims[0] = dim1;
            dims[1] = dim2;

            auto size = varProduct(dim1, dim2);
            allocValues(size, d);
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, T dim3, Device d=Device::CPU)
            : device(d) { 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);

            dims[0] = dim1;
            dims[1] = dim2;
            dims[2] = dim3;

            auto size = varProduct(dim1, dim2, dim3);
            allocValues(size, d);
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, T dim3, T dim4, Device d=Device::CPU)
            : device(d) { 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);
            assert(dim4 >= 0);

            dims[0] = dim1; 
            dims[1] = dim2; 
            dims[2] = dim3; 
            dims[3] = dim4;

            auto size = varProduct(dim1, dim2, dim3, dim4);
            allocValues(size, d);
        }

        const Dimension& getDims() const noexcept;
        Tensor operator*(Tensor const& t);
};