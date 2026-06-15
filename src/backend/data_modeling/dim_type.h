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

#include "shared/global_params.h"

#include "utility/utils.h"

#include <vector>
#include <array>
#include <memory>

#include <iostream>
#include <cassert>

class Dimension final {
  using dim_t = std::array<tensorSize_t, MAX_NDIMS>;

  private:
    // those two indicate the structure of the contiguous data that lies underneath
    std::shared_ptr< std::vector<tensorDim_t> > contiguousDims;
    std::shared_ptr<dim_t> contiguousStrides;

    std::vector<tensorDim_t> dims;
    dim_t strides;

    tensorDim_t lastDimIdx; // look up end in strides/dims
    tensorSize_t size = 0; // total size of tensor

    dim_t makeStrides(const std::vector<tensorDim_t>& dims) const noexcept;
    tensorSize_t multVector(const std::vector<tensorDim_t>& dims) const noexcept;
    
    Dimension(std::vector<tensorDim_t>&& dims, dim_t&& strides);

    tensorDim_t mapSignedIdx(int idx) const {
      if(idx < 0) {
        // -1 is last idx, -2 second last and so forth
        return lastDimIdx + idx + 1;
      }
      return idx;
    }

  public:
    Dimension(const std::vector<tensorDim_t>& dims);

    Dimension(const Dimension& other) = default;
    Dimension& operator=(const Dimension& other) = default;

    Dimension(Dimension&& other) noexcept = default;
    Dimension& operator=(Dimension&& other) noexcept = default;

    ~Dimension() noexcept = default;

    Dimension collapseDimension(int idx) const;

    bool inOriginalState() const noexcept {
      assert_debug((*contiguousDims == dims && contiguousStrides == strides) ||
                   (*contiguousDims != dims && contiguousStrides != strides),
                   "Swapping dims implies swapping strides");
      return *contiguousDims == dims; 
    }

    void resize(const std::vector<tensorDim_t>& dims);
      
    tensorSize_t getSize() const noexcept {
      return size;
    }

    tensorDim_t get(int idx) const {
      return (*this)[idx];
    }

    tensorDim_t operator[](int idx) const {
      assert(size > 0);
      return dims[mapSignedIdx(idx)];
    }

    const tensorDim_t* data() const noexcept { return dims.data(); }
    
    const auto getStrides() const noexcept { return strides; }
    const auto getContiguousStrides() const noexcept { return contiguousStrides; }
    tensorSize_t getStride(int i) const noexcept;

    std::vector<tensorDim_t> toVector() const noexcept{
      return dims;
    }

    void swap(int dim1, int dim2);

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