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

#include <vector>

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
    std::vector<tensorDim_t> dims;
    tensorSize_t size = 0;

    tensorDim_t multVector(const std::vector<tensorDim_t>& dims) const noexcept;

  public:
    /**
     * @brief Explicit default ctor, so that dims is zero initialized.
     * Otherwise we will encounter undefined behavior.
     */
    Dimension(const std::vector<tensorDim_t>& dims);

    Dimension(const Dimension& other);
    Dimension& operator=(const Dimension& other);

    Dimension(Dimension&& other) noexcept;
    Dimension& operator=(Dimension&& other) noexcept;

    ~Dimension() noexcept = default;

    void resize(const std::vector<tensorDim_t>& dims);
    tensorSize_t getSize() const noexcept {
      assert(size!=0);
      return size;
    }

    tensorDim_t get(int idx) const {
      assert(size!=0);
      if(idx<0){
        idx = dims.size() + idx; // -1 is last idx, -2 second last and so forth
      }

      return dims[idx];
    }

    std::vector<tensorDim_t> toVector() const noexcept{
      return dims;
    }

    void swap(const tensorDim_t dim1, const tensorDim_t dim2);

    size_t nDims() const noexcept {
      assert(size!=0);
      return dims.size();
    }

    bool operator==(const Dimension& other) const {
      assert(size!=0);
      return this->dims == other.dims;
    }

    bool operator!=(const Dimension& other) const {
      return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os, const Dimension& d) noexcept;
};