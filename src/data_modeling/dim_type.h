/**
 * @file dim_type.h
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
#include <cstdint>
#include <cassert>

template <typename T>
concept is_valid_dim = requires(T x) {
    requires std::is_integral_v<std::remove_const_t<T>>;
    requires std::convertible_to<std::remove_const_t<T>, std::uint16_t>;
    x >= 0;
};

class Dimension final {
  private:
    std::array<std::uint16_t, MAX_TENSOR_SIZE> dims; // assumption: maximum dimension of Tensor is 4

  public:
    std::uint16_t& operator[](int idx){
      assert(idx < MAX_TENSOR_SIZE);
      return dims[idx];
    }

    bool operator==(const Dimension& other) const {
      return this->dims == other.dims;
    }
};