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
#include "device.h"

#include "computational_graph/topological_sort.h"
#include "computational_graph/graph_node.h"

#include "shared/global_params.h"
#include "shared/initializers.h"

#include <memory>
#include <span>

#include <iostream>

#include <concepts>
#include <type_traits>
#include <cassert>

// break circular dependency
namespace cgraph
{
  class GraphNode;
  class TopologicalSort;
}

class Tensor final : public std::enable_shared_from_this<Tensor>
{
  friend class cgraph::TopologicalSort;

private:
  struct shallowCopyToken{}; // we only want createShallowCopy to do shallow copies

  /**
   * @brief Here we encapsulate the tensor's values.
   * Enables us to use a shared_ptr, as well as encapsulate all the
   * memory management logic for different devices, like a GPU.
   * Structured as a flat array, the logic for multiple dimensions
   * encapsulated by surrounding tensor object.
   */
  class tensorValues_t final
  {
  private:
    tensorSize_t size = 0;
    ftype* values = nullptr;

    Device device;
    inline static Device defaultDevice = Device::CPU;

  public:
    explicit tensorValues_t();
    explicit tensorValues_t(Device d);
    ~tensorValues_t() noexcept;

    tensorValues_t(const tensorValues_t& other) = delete;
    tensorValues_t& operator=(const tensorValues_t& other) = delete;

    tensorValues_t(tensorValues_t&& other) noexcept;
    tensorValues_t& operator=(tensorValues_t&& other) noexcept;

    void copyFromRaw(const ftype* src, tensorSize_t n);

    ftype* data() noexcept { return values; }
    const ftype* data() const noexcept { return values; }
    
    explicit operator bool() const noexcept;

    ftype operator[](tensorSize_t idx) const;
    void set(ftype v, tensorSize_t idx);
    ftype get(tensorSize_t idx);

    tensorSize_t getSize() const noexcept;

    void resize(tensorSize_t size);

    void setDevice(Device d) noexcept;
    Device getDevice() const noexcept;


    void copyValues(tensorValues_t& target) const;
    void copyValues(tensorValues_t& target, tensorSize_t low, tensorSize_t high, tensorSize_t targetOffset) const;

    static void setDefaultDevice(Device d) noexcept;
    static Device getDefaultDevice() noexcept;
  };

  mutable Dimension dims;
  mutable std::shared_ptr<tensorValues_t> values = nullptr; // contained values of tensor

  bool requiresGrad = false;
  std::shared_ptr<Tensor> grads = nullptr; // gradients
  std::shared_ptr<cgraph::GraphNode> cgNode = nullptr;

  static Tensor matMulImpl(const Tensor& left, const Tensor& right, bool transposeLeft, bool transposeRight);
  
  template<bool transposeLeft, bool transposeRight>
  static void matMul2DCpuScalar(Tensor& res, const Tensor& left, const Tensor& right,
                          tensorSize_t resOffset, tensorSize_t leftOffset,
                          tensorSize_t rightOffset);
  
                          template<bool transposeLeft, bool transposeRight>
  static void matMul2DCpuAvx(Tensor& res, const Tensor& left, const Tensor& right,
                          tensorSize_t resOffset, tensorSize_t leftOffset,
                          tensorSize_t rightOffset);

  void makeContiguous() const;

  // convenience functions that appear in multiple places
  static tensorSize_t computeLinearIdx(const std::vector<tensorDim_t>&& idx, const Dimension& dims);
  static tensorSize_t computeLinearIdx(const std::vector<tensorDim_t>& idx, const Dimension& dims);

  static tensorDim_t mapDim(int dim, const Dimension& dims);

  Tensor(const Tensor& other, shallowCopyToken);

public:
  template <typename T>
    requires(std::is_same_v<std::remove_cvref_t<T>, Dimension>)
  explicit Tensor(T&& dims, Device d, bool requiresGrad = false)
      : Tensor{dims.toVector(), d, requiresGrad}
      // !!!needs dims.toVector() to not trigger the copy ctors!!!
  { }

  explicit Tensor(const std::vector<tensorDim_t>& dims, bool requiresGrad = false) 
    : dims{dims}, values{std::make_unique<tensorValues_t>()}, requiresGrad{requiresGrad}
  {
    values->resize(this->dims.getSize());
  }

  explicit Tensor(const std::vector<tensorDim_t>& dims, Device d, bool requiresGrad = false) 
    : dims{dims}, values{std::make_unique<tensorValues_t>(d)}, requiresGrad{requiresGrad}
  {
    values->resize(this->dims.getSize());
  }

  explicit Tensor(const std::vector<tensorDim_t>& dims, const std::vector<ftype>& initValues, bool requiresGrad = false) 
    : Tensor{dims, std::move(initValues), Tensor::getDefaultDevice(), requiresGrad}
  {
  }

  explicit Tensor(const std::vector<tensorDim_t>& dims, const std::vector<ftype>& initValues, Device d, bool requiresGrad = false) 
    : Tensor{dims, d, requiresGrad}
  {
    for (tensorSize_t i=0; i<initValues.size(); i++){
      values->set(initValues[i], i);
    }
  }

  Tensor(const std::vector<tensorDim_t>& dims, const ftype *data, tensorSize_t dataSize,
         Device d = Device::CPU, bool requiresGrad = false)
      : Tensor(dims, d, requiresGrad)
  {
    assert(values->getSize() == dataSize);
    values->copyFromRaw(data, dataSize);
  }

  /**
   * Tensors can become very large. Deleting those two
   * helps us to not accidentally copy something we do not
   * intend to copy. We create extra methods for that.
   */
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor createEmptyCopy() const;
  Tensor createShallowCopy() const;
  Tensor createContiguousCopy() const;
  Tensor createDeepCopy() const;

  /**
   * @brief Moving. Move array of values
   * to new instance as well.
   */
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  ftype* getData() const noexcept;

  void reset(ftype x) noexcept;
  void reset(std::shared_ptr<utility::InitializerBase> init) noexcept;

  const Dimension& getDims() const noexcept;
  tensorSize_t getSize() const noexcept;

  // Tensor operator@(const Tensor& other) const; in higher C++ versions than 20
  Tensor matmul(const Tensor& other, bool transposeLeft=false, bool transposeRight=false) const;

  Tensor operator+(const Tensor& other) const;
  Tensor add(const Tensor& other) const;

  // TODO: Tensor operator-(const Tensor& other) const;

  Tensor operator*(const Tensor& t) const;
  Tensor elementwiseMul(const Tensor& other) const;

  Tensor& operator+=(const Tensor& other);

  // TODO: Tensor operator/(const Tensor& other) const;

  // the following operators all broadcast
  Tensor operator*(ftype scalar) const;
  Tensor operator/(ftype scalar) const;
  Tensor operator+(ftype scalar) const;
  Tensor operator-(ftype scalar) const;

  // turn around the arguments as well: scalar *:+ tensor
  friend Tensor operator*(ftype scalar, const Tensor& tensor);
  friend Tensor operator+(ftype scalar, const Tensor& tensor);

  void backward();

  std::shared_ptr<Tensor> getGrads() const;
  void setGrads(std::shared_ptr<Tensor> grads) noexcept{
    this->grads = std::move(grads);
  }
  bool hasGrads() const noexcept { return grads!=nullptr; }

  Tensor transpose(int dim1=-1, int dim2=-2);
  void permute(const std::vector<tensorDim_t>& newOrder) noexcept;

  bool isContiguous() const noexcept { return dims.inOriginalState(); }
  Tensor getContiguous() const;

  friend std::ostream& operator<<(std::ostream& os, const Tensor& t) noexcept;

  // for convenience we provide some simple getters
  ftype get(tensorSize_t idx) const;
  ftype get(tensorDim_t idx0, tensorDim_t idx1) const;
  ftype get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) const;
  ftype get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) const;

  // non-const version of operator[] does not exist because of CUDA
  ftype operator[](tensorSize_t idx) const;

  ftype get(const std::vector<tensorDim_t>& idx) const;

  // for convenience we provide some simple setters
  void set(ftype item, tensorDim_t idx);
  void set(ftype item, tensorDim_t idx0, tensorDim_t idx1);
  void set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2);
  void set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3);
  void set(ftype item, const std::vector<tensorDim_t>& idx);

  void setDevice(const Device d) noexcept;
  Device getDevice() const noexcept;

  bool getRequiresGrad() const noexcept { return requiresGrad; }
  void setRequiresGrad(const bool requiresGrad) noexcept { this->requiresGrad=requiresGrad; }

  void setCgNode(std::shared_ptr<cgraph::GraphNode> node) noexcept {
    cgNode = std::move(node);
    requiresGrad = true;
  }

  std::shared_ptr<Tensor> getSharedPtr() const
  {
    try{
      return std::const_pointer_cast<Tensor>(shared_from_this());
    }
    catch (const std::bad_weak_ptr&) {
      throw std::runtime_error(
          "Tensor must be managed by shared_ptr for autograd operations");
    }
  }

  Tensor getSlice(tensorSize_t low, tensorSize_t high) const;
  Tensor getSlice(std::span<const tensorDim_t> indices) const;

  // these two should not be exposed to the python interface
  static void setDefaultDevice(const Device d) noexcept;
  static Device getDefaultDevice() noexcept;
};

/**
 * @brief Name says it all. Inplace operation on res.
 * 
 * Transposition baked into kernel. Assumption is that kernels are not transposed, but we treat them 
 * as if they were for increased speed without the actual transposition happening.
 */
template<bool transposeLeft, bool transposeRight>
void Tensor::matMul2DCpuScalar(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset, 
                           const tensorSize_t leftOffset, const tensorSize_t rightOffset) {
  // physical, not logically as in the transposition
  const auto nRowsLeft = static_cast<tensorSize_t>(left.dims.get(-2));
  const auto nColsLeft = static_cast<tensorSize_t>(left.dims.get(-1));
  const auto nColsRight = static_cast<tensorSize_t>(right.dims.get(-1));
  const auto nRowsRight = static_cast<tensorSize_t>(right.dims.get(-2));

  if constexpr (!transposeLeft && !transposeRight) {
    for (tensorSize_t i = 0; i < nRowsLeft; i++) {
      const tensorSize_t leftOffset = i * nColsLeft;
      const tensorSize_t resOffset = i * nColsRight;

      {
        // tensors not zero initialized, therefore need one dry run pre-filling
        const auto leftVal = left[leftOffset];
        for(tensorSize_t j = 0; j < nColsRight; j++) {
          res.values->data()[resOffset + j] = leftVal * right[j];
        }
      }

      // compute the entire column of the left matrix
      for (tensorSize_t k = 1; k < nRowsRight; k++) {
        const tensorSize_t rightOffset = k * nColsRight;

        const ftype leftVal = left[leftOffset + k];
        for(tensorSize_t j = 0; j < nColsRight; j++) {
          res.values->data()[resOffset + j] += leftVal * right[rightOffset + j];
        }
      }
    }
  }
  else if constexpr (!transposeLeft && transposeRight) {
    for (tensorSize_t i = 0; i < nRowsLeft; i++) {
      const tensorSize_t leftOffset = i * nColsLeft;
      const tensorSize_t resOffset = i * nRowsRight;

      for(tensorSize_t j = 0; j < nRowsRight; j++) {
        const tensorSize_t rightOffset = j * nColsRight;

        ftype sum = 0.0f;
        for (tensorSize_t k = 0; k < nColsRight; k++) { 
          sum += left[leftOffset + k] * right[rightOffset + k];
        }

        res.values->data()[resOffset + j] = sum;
      }
    }
  }
  else if constexpr (transposeLeft && !transposeRight) {
    // initialize whole res matrix
    for (tensorSize_t i = 0; i < nColsLeft; i++) {
      const tensorSize_t resOffset = i * nColsRight; // start of row

      // tensors not zero initialized, therefore need one dry run pre-filling
      const ftype leftVal = left[i];
      for(tensorSize_t j = 0; j < nColsRight; j++) {
        res.values->data()[resOffset + j] = leftVal * right[j];
      }
    }

    for(tensorSize_t k = 1; k < nRowsLeft; k++) {
      const tensorSize_t leftOffset = k * nColsLeft;
      const tensorSize_t rightOffset = k * nColsRight;

      for (tensorSize_t i = 0; i < nColsLeft; i++) {
        const tensorSize_t resOffset = i * nColsRight;

        const ftype leftVal = left[leftOffset + i];
        for(tensorSize_t j = 0; j < nColsRight; j++) {
          res.values->data()[resOffset + j] += leftVal * right[rightOffset + j];
        }
      }
    }
  }
  else {
    for (tensorSize_t i = 0; i < nColsLeft; i++) {
      const tensorSize_t resOffset = i * nRowsRight;

      for (tensorSize_t j = 0; j < nRowsRight; j++) {
        const tensorSize_t rightOffset = j * nColsRight;

        ftype sum = 0.0f;
        for (tensorSize_t k = 0; k < nRowsLeft; k++) {
          sum += left[k * nColsLeft + i] * right[rightOffset + k];
        }

        res.values->data()[resOffset + j] = sum;
      }
    }
  }
}

/**
 * @brief Name says it all. Inplace operation on res.
 */
/* template<bool transposeLeft, bool transposeRight>
void Tensor::matMul2DCpuAvx(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset, 
                           const tensorSize_t leftOffset, const tensorSize_t rightOffset) {
  
  const auto nRowsLeft = static_cast<tensorSize_t>(left.dims.get(-2));
  const auto nColsLeft = static_cast<tensorSize_t>(left.dims.get(-1));
  const auto nColsRight = static_cast<tensorSize_t>(right.dims.get(-1));
  const auto nRowsRight = static_cast<tensorSize_t>(right.dims.get(-2));

  const tensorSize_t M = transposeLeft ? nColsLeft : nRowsLeft;
  const tensorSize_t K = transposeLeft ? nRowsLeft : nColsLeft;
  const tensorSize_t N = transposeRight ? nRowsRight : nColsRight;

  for (tensorSize_t i = 0; i < M; i++) {
    for (tensorSize_t j = 0; j < N; j++) {
      ftype sum = 0.0f;

      for (tensorSize_t k = 0; k < K; k++) {
        tensorSize_t leftIdx = transposeLeft ? leftOffset + k * nColsLeft + i
                                             : leftOffset + i * nColsLeft + k;
        
        tensorSize_t rightIdx = transposeRight ? rightOffset + j * nColsRight + k
                                               : rightOffset + k * nColsRight + j;

      #if defined(USE_AVX512)
        
      #elif defined(USE_AVX2)
        
      #elif defined(USE_AVX)
        
      #else
        std::__throw_runtime_error("Not compiled with AVX enabled");
      #endif

        sum += left.values->data()[leftIdx] * right.values->data()[rightIdx];
      }

      res.values->data()[resOffset + i * N + j] = sum;
    }
  }
} */