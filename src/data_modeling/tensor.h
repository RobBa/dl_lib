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

#include "global_params.h"

#include <array>
#include <concepts>

#include <cstdint>

namespace {
    template <typename T>
    concept is_valid_dim = requires(T x) {
        requires std::is_integral_v<T>;
        requires std::convertible_to<T, std::uint16_t>;
        x >= 0;
    };
}

struct Tensor final {
    private:
        ftype* values = nullptr;
        std::array<std::uint16_t, 4> dims{0, 0, 0, 0}; // assumption: maximum dimension of Tensor is 4

    public:
        Tensor() = default;
        
        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1){
            assert(dim1 >= 0);
            
            dims[0] = dim1; 
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2){
            assert(dim1 >= 0);
            assert(dim2 >= 0);

            dims[0] = dim1;
            dims[1] = dim2;
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, T dim3){ 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);

            dims[0] = dim1;
            dims[1] = dim2;
            dims[2] = dim3;
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, T dim3, T dim4){ 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);
            assert(dim4 >= 0);

            dims[0] = dim1; 
            dims[1] = dim2; 
            dims[2] = dim3; 
            dims[3] = dim4; 
        }

        const std::array<std::uint16_t, 4>& getDims() const noexcept;
};