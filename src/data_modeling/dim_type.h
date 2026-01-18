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

#include <iostream>
#include <cassert>

template <typename T>
concept is_valid_dim = requires(T x) {
    requires std::is_integral_v<std::remove_const_t<T>>;
    requires std::convertible_to<std::remove_const_t<T>, tensorDim_t>;
    x >= 0;
};

class Dimension final {
  private:
    std::array<tensorDim_t, MAX_TENSOR_DIMS> dims; // assumption: maximum dimension of Tensor is 4

  public:
    tensorDim_t& operator[](int idx){
      assert(idx < MAX_TENSOR_DIMS);
      return dims[idx];
    }

    tensorDim_t get(int idx) const {
      assert(idx < MAX_TENSOR_DIMS);
      return dims[idx];
    }

    bool operator==(const Dimension& other) const {
      return this->dims == other.dims;
    }

    tensorSize_t getTotalSize() const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const Dimension& d) noexcept;
};