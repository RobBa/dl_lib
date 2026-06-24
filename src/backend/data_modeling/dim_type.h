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
#include "utility/safe_arithmetics.h"

#include <vector>
#include <array>
#include <memory>
#include <utility>

#include <iostream>
#include <cassert>

class Dimension final {
  using dim_t = std::array<tensorSize_t, MAX_NDIMS>;
  struct shallowCopyToken {};

  private:
    // those two indicate the structure of the contiguous data that lies underneath
    std::shared_ptr< std::vector<tensorDim_t> > contiguousDims;
    std::shared_ptr<dim_t> contiguousStrides;

    std::vector<tensorDim_t> dims;
    dim_t strides;

    int lastDimIdx; // look up end in strides/dims
    tensorSize_t size = 0; // total size of tensor

    tensorSize_t multVector(const std::vector<tensorDim_t>& dims) const noexcept {
      tensorSize_t res = 1;
#ifndef NDEBUG
      utility::SafeArithmetics_t<tensorSize_t> mult(1);
      for(auto dim : dims)
        mult = mult * dim;
      res = mult.value;
#else
      for(auto dim : dims)
        res *= dim;
#endif
      return res;
    }

    dim_t makeStrides(const std::vector<tensorDim_t>& dims) const noexcept {
      dim_t res;
      const int lastDimIdx = dims.size() - 1;
      tensorSize_t stride = 1;
      res[lastDimIdx] = stride;
      stride *= dims[lastDimIdx];
      for(int i = lastDimIdx - 1; i >= 0; i--) {
        res[i] = stride;
        stride *= dims[i];
      }
      return res;
    }

    Dimension(std::vector<tensorDim_t>&& dims, dim_t&& strides);

    Dimension(const Dimension& other, shallowCopyToken)
      : dims{other.dims},
        strides{other.strides},
        contiguousDims{other.contiguousDims},
        contiguousStrides{other.contiguousStrides},
        lastDimIdx{other.lastDimIdx},
        size{other.size}
    {}

    int mapSignedIdx(int idx) const {
      if(idx < 0) {
        // -1 is last idx, -2 second last and so forth
        return lastDimIdx + (idx + 1);
      }
      return idx;
    }

  public:
    Dimension(const std::vector<tensorDim_t>& dims)
      : contiguousDims{std::make_shared<std::vector<tensorDim_t>>(dims)},
        contiguousStrides{std::make_shared<dim_t>(makeStrides(dims))},
        dims{dims},
        strides{*contiguousStrides}
    {
      size = multVector(dims);
      lastDimIdx = dims.size() - 1;
      assert(size > 0);
    }

    Dimension(const Dimension& other) {
      dims = other.dims;
      strides = other.strides;
      contiguousDims = std::make_shared<std::vector<tensorDim_t>>(*other.contiguousDims);
      contiguousStrides = std::make_shared<dim_t>(*other.contiguousStrides);
      lastDimIdx = other.lastDimIdx;
      size = other.size;
    }

    Dimension& operator=(const Dimension& other) {
      if(&other == this) return *this;
      dims = other.dims;
      strides = other.strides;
      contiguousDims = std::make_shared<std::vector<tensorDim_t>>(*other.contiguousDims);
      contiguousStrides = std::make_shared<dim_t>(*other.contiguousStrides);
      lastDimIdx = other.lastDimIdx;
      size = other.size;
      return *this;
    }

    Dimension(Dimension&& other) noexcept = default;
    Dimension& operator=(Dimension&& other) noexcept = default;

    ~Dimension() noexcept = default;

    Dimension shallowCopy() const { return Dimension(*this, shallowCopyToken{}); }

    Dimension collapseDimension(int idx) const;

    /**
     * @brief For quadratic tensors the stride changes upon a transpose,
     * but the dims remain intact. Hence stride is only reliable source of
     * memory layout deductions.
     */
    bool inOriginalState() const noexcept {
      return *contiguousStrides == strides;
    }

    void resize(const std::vector<tensorDim_t>& dims) {
      this->dims = dims;
      size = multVector(dims);
      assert(size > 0);
    }

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

    tensorSize_t getStride(int i) const noexcept {
      if(i < 0)
        return strides[lastDimIdx + i + 1];
      return strides[i];
    }

    void makeContiguous() {
      contiguousDims = std::make_shared<std::vector<tensorDim_t>>(dims);
      contiguousStrides = std::make_shared<dim_t>(strides);
    }

    std::vector<tensorDim_t> toVector() const noexcept {
      return dims;
    }

    void swap(int dim1, int dim2) {
      if(dim1 == dim2) return;
      auto d1 = mapSignedIdx(dim1);
      auto d2 = mapSignedIdx(dim2);
      assert(d1 >= 0);
      assert(d2 >= 0);
      std::swap(dims[d1], dims[d2]);
      std::swap(strides[d1], strides[d2]);
    }

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
