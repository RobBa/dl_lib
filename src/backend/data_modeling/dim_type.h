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

#include "utility/global_params.h"

#include <vector>
#include <array>

#include <iostream>
#include <cassert>

class Dimension final {
  private:
    const std::vector<tensorDim_t> creationDims;
    const std::array<tensorSize_t, MAX_NDIMS> creationStrides;

    std::vector<tensorDim_t> dims;
    std::array<tensorSize_t, MAX_NDIMS> strides;

    tensorDim_t lastDimIdx; // look up end in strides/dims
    tensorSize_t size = 0; // total size of tensor

    std::array<tensorSize_t, MAX_NDIMS> Dimension::createStrides(const std::vector<tensorDim_t>& dims) const noexcept;
    tensorSize_t multVector(const std::vector<tensorDim_t>& dims) const noexcept;
    
    Dimension(std::vector<tensorDim_t>&& dims, std::array<tensorSize_t, MAX_NDIMS>&& strides);

  public:

    Dimension(const std::vector<tensorDim_t>& dims);

    Dimension(const Dimension& other);
    Dimension& operator=(const Dimension& other) = delete;

    Dimension(Dimension&& other) noexcept;
    Dimension& operator=(Dimension&& other) noexcept = delete;

    ~Dimension() noexcept = default;

    Dimension collapseDimension(int idx) const;

    void resize(const std::vector<tensorDim_t>& dims);
      
    tensorSize_t getSize() const noexcept {
      return size;
    }

    tensorDim_t get(int idx) const {
      return (*this)[idx];
    }

    tensorDim_t operator[](int idx) const {
      assert(size>0);
      if(idx<0){
        return dims[lastDimIdx + idx + 1]; // -1 is last idx, -2 second last and so forth
      }

      return dims[idx];
    }

    tensorSize_t getStride(int i) const noexcept;

    std::vector<tensorDim_t> toVector() const noexcept{
      return dims;
    }

    void swap(tensorDim_t dim1, tensorDim_t dim2);

    size_t nDims() const noexcept {
      return dims.size();
    }

    bool operator==(const Dimension& other) const {
      assert(size!=0);
      return this->dims == other.dims;
    }

    bool operator==(const std::vector<tensorDim_t>& other) const {
      return this->dims == other;
    }

    bool operator!=(const Dimension& other) const {
      return !(*this == other);
    }

    bool operator!=(const std::vector<tensorDim_t>& other) const {
      return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os, const Dimension& d) noexcept;    
};