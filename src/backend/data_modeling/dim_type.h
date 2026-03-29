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
  using dim_t = std::array<tensorSize_t, MAX_NDIMS>;

  private:
    // creationDims and creationStrides should only be set in constructors, otherwise
    // bugs will emerge. Made non-const so we can use move-assignment operator
    std::vector<tensorDim_t> creationDims;
    dim_t creationStrides;

    std::vector<tensorDim_t> dims;
    dim_t strides;

    tensorDim_t lastDimIdx; // look up end in strides/dims
    tensorSize_t size = 0; // total size of tensor

    dim_t Dimension::makeStrides(const std::vector<tensorDim_t>& dims) const noexcept;
    tensorSize_t multVector(const std::vector<tensorDim_t>& dims) const noexcept;
    
    Dimension(std::vector<tensorDim_t>&& dims, dim_t&& strides);

  public:

    Dimension(const std::vector<tensorDim_t>& dims);

    Dimension(const Dimension& other);
    Dimension& operator=(const Dimension& other);

    Dimension(Dimension&& other) noexcept;
    Dimension& operator=(Dimension&& other) noexcept;

    ~Dimension() noexcept = default;

    Dimension collapseDimension(int idx) const;

    bool inOriginalState() const noexcept { 
      return creationDims == dims && creationStrides == strides; 
    }

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

    const auto getStrides() const noexcept { return strides; }
    const auto getOriginalStrides() const noexcept { return creationStrides; }

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