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

#include <iostream>
#include <cassert>

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

    Dimension collapseDimension(int idx) const;

    void resize(const std::vector<tensorDim_t>& dims);
      
    tensorSize_t getSize() const noexcept {
      return size;
    }

    tensorDim_t getItem(int idx) const {
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
      return dims.size();
    }

    /**
     * @brief Returns empty dims. Used e.g. to identify dimensions
     * of activation functions.
     */
    static const Dimension& getEmpty() {
      static const auto emptyDims = Dimension(std::vector<tensorDim_t>());
      return emptyDims;
    }

    bool empty() const noexcept {
      return size > 0;
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