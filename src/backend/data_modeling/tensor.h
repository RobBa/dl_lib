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
#include "matmul_tile.h"

#include "computational_graph/topological_sort.h"
#include "computational_graph/graph_node.h"

#include "shared/global_params.h"
#include "shared/initializers.h"
#include "shared/memory_pool.h"
#include "utility/memory_layout.h"

#include <memory>
#include <span>

#include <iostream>

#include <concepts>
#include <type_traits>
#include <cassert>

#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX)
#include <immintrin.h>
#endif

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
  struct shallowCopyToken{};

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
    explicit tensorValues_t() { device = defaultDevice; }
    explicit tensorValues_t(Device d) : device(d) {}

    ~tensorValues_t() noexcept {
      if(values != nullptr)
        mempool::tensorPool.giveback(values, device, size);
    }

    tensorValues_t(const tensorValues_t& other) = delete;
    tensorValues_t& operator=(const tensorValues_t& other) = delete;

    tensorValues_t(tensorValues_t&& other) noexcept
      : device{std::move(other.device)}, size{std::move(other.size)}, values{std::move(other.values)}
    {
      // ensuring destructor does not return pointer to memory pool
      other.values = nullptr;
      other.size = 0;
    }

    tensorValues_t& operator=(tensorValues_t&& other) noexcept {
      if(this == &other) return *this;
      device = std::move(other.device);
      size = std::move(other.size);
      values = std::move(other.values);
      // ensuring destructor does not return pointer to memory pool
      other.values = nullptr;
      other.size = 0;
      return *this;
    }

    void copyFromRaw(const ftype* src, tensorSize_t n);

    ftype* data() noexcept { return values; }
    const ftype* data() const noexcept { return values; }

    explicit operator bool() const noexcept { return values != nullptr; }

    ftype operator[](tensorSize_t idx) const;
    void set(ftype v, tensorSize_t idx);
    ftype get(tensorSize_t idx);

    tensorSize_t getSize() const noexcept { return size; }

    void resize(tensorSize_t size);

    void setDevice(Device d) noexcept;
    Device getDevice() const noexcept { return device; }

    void copyValues(tensorValues_t& target) const;
    void copyValues(tensorValues_t& target, tensorSize_t low, tensorSize_t high, tensorSize_t targetOffset) const;

    static void setDefaultDevice(Device d) noexcept { defaultDevice = d; }
    static Device getDefaultDevice() noexcept { return defaultDevice; }
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

#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX)
  template<bool transposeLeft, bool transposeRight>
  static void matMul2DCpuAvx(Tensor& res, const Tensor& left, const Tensor& right,
                          tensorSize_t resOffset, tensorSize_t leftOffset,
                          tensorSize_t rightOffset);
#endif

  void makeContiguous() const;

  /**
   * @brief Computes the 1D index from a set of indices.
   *
   * WARNING: Does not check for overflow.
   */
  static tensorSize_t computeLinearIdx(const std::vector<tensorDim_t>& idx, const Dimension& dims) {
    #ifndef NDEBUG
      if(idx.size() != dims.nDims()) {
        std::__throw_invalid_argument("Number of idxs must match number of dimensions.");
      }
      else if(idx.size() == 0){
        return 0;
      }
    #endif

    tensorSize_t res = 0;
    for(tensorDim_t i = 0; i < idx.size(); i++){
      res += idx[i] * dims.getStride(i);
    }
    return res;
  }

  static tensorSize_t computeLinearIdx(const std::vector<tensorDim_t>&& idx, const Dimension& dims) {
    return computeLinearIdx(idx, dims);
  }

  static tensorDim_t mapDim(int dim, const Dimension& dims) {
    if(dim >= 0) return dim;
    else if(dim + (int)dims.nDims() < 0) std::__throw_invalid_argument("Invalid dim value given.");
    return dims.nDims() + dim;
  }

  Tensor(const Tensor& other, shallowCopyToken)
    : dims(other.dims.shallowCopy()), cgNode{other.cgNode}, values{other.values},
      grads{other.grads}, requiresGrad{other.requiresGrad}
  {}

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

  Tensor createEmptyCopy() const { return Tensor(dims, values->getDevice(), requiresGrad); }
  Tensor createShallowCopy() const { return Tensor(*this, shallowCopyToken{}); }
  Tensor createContiguousCopy() const;
  Tensor createDeepCopy() const;

  /**
   * @brief Moving. Move array of values to new instance as well.
   */
  Tensor(Tensor&& other) noexcept
    : dims{std::move(other.dims)},
      values{std::move(other.values)},
      requiresGrad{other.requiresGrad},
      cgNode{std::move(other.cgNode)},
      grads{std::move(other.grads)}
  {}

  Tensor& operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    dims = std::move(other.dims);
    values = std::move(other.values);
    requiresGrad = other.requiresGrad;
    
    cgNode = std::move(other.cgNode);
    grads = std::move(other.grads);

    return *this;
  }

  ftype* getData() const noexcept { return values->data(); }

  void reset(ftype x) noexcept;
  void reset(std::shared_ptr<utility::InitializerBase> init) noexcept;

  const Dimension& getDims() const noexcept { return dims; }
  tensorSize_t getSize() const noexcept { return values->getSize(); }

  /**
   * @brief Matrix multiplication. Transpose flags are optimizations to avoid a
   * physical transpose in memory before the computation. Transposition done on the last two
   * dimensions, the dimensions the matmul is executed on.
   * 
   * TODO: Tensor operator@(const Tensor& other) const; in higher C++ versions than 20
   */
  Tensor matmul(const Tensor& other, bool transposeLeft = false, bool transposeRight = false) const {
  #ifndef NDEBUG
    if(values->getDevice() != other.values->getDevice()){
      std::__throw_runtime_error("Tensors on different devices.");
    }
  #endif

    return matMulImpl(getContiguous(), other.getContiguous(), transposeLeft, transposeRight);
  }

  Tensor operator+(const Tensor& other) const;
  Tensor add(const Tensor& other) const { return *this + other; }

  // TODO: Tensor operator-(const Tensor& other) const;

  Tensor operator*(const Tensor& t) const;
  Tensor elementwiseMul(const Tensor& other) const { return *this * other; }

  Tensor& operator+=(const Tensor& other);

  // TODO: Tensor operator/(const Tensor& other) const;

  // the following operators all broadcast
  Tensor operator*(ftype scalar) const;
  Tensor operator/(ftype scalar) const;
  Tensor operator+(ftype scalar) const;
  Tensor operator-(ftype scalar) const;

  // turn around the arguments as well: scalar *:+ tensor
  friend Tensor operator*(ftype scalar, const Tensor& tensor) { return tensor * scalar; }
  friend Tensor operator+(ftype scalar, const Tensor& tensor) { return tensor + scalar; }

  void backward();

  std::shared_ptr<Tensor> getGrads() const { return grads; }
  void setGrads(std::shared_ptr<Tensor> grads) noexcept {
    this->grads = std::move(grads);
  }
  bool hasGrads() const noexcept { return grads!=nullptr; }

  Tensor transpose(int dim1=-1, int dim2=-2) {
    Tensor result = createShallowCopy();
    result.dims.swap(dim1, dim2);
    return result;
  }
  void permute(const std::vector<tensorDim_t>& newOrder) noexcept {
    assert(newOrder.size()==dims.nDims());
    for(tensorDim_t i=0; i<static_cast<tensorDim_t>(newOrder.size()); i++)
      dims.swap(i, newOrder[i]);
  }

  bool isContiguous() const noexcept { return dims.inOriginalState(); }
  Tensor getContiguous() const {
    if(dims.inOriginalState()) return createShallowCopy();
    return createContiguousCopy();
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& t) noexcept;

  // for convenience we provide some simple getters
  ftype get(const std::vector<tensorDim_t>& idx) const {
    makeContiguous();
    return (*values)[computeLinearIdx(idx, dims)];
  }
  ftype get(tensorDim_t idx0, tensorDim_t idx1) const { return get({idx0, idx1}); }
  ftype get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) const { return get({idx0, idx1, idx2}); }
  ftype get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) const { return get({idx0, idx1, idx2, idx3}); }

  // non-const version of operator[] does not exist because of CUDA
  ftype operator[](tensorSize_t idx) const { return (*values)[idx]; }

  /**
   * @brief Special getter, indexes the contained underlying array linearly.
   * Can lead to unexpected results in multidimensional tensors.
   */
  ftype get(tensorSize_t idx) const { return (*this)[idx]; }

  // for convenience we provide some simple setters
  void set(ftype item, tensorDim_t idx0, tensorDim_t idx1) { set(item, {idx0, idx1}); }
  void set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) { set(item, {idx0, idx1, idx2}); }
  void set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) { set(item, {idx0, idx1, idx2, idx3}); }
  void set(ftype item, const std::vector<tensorDim_t>& idx) {
    makeContiguous();
    values->set(item, computeLinearIdx(idx, dims));
  }

  /**
   * @brief Special setter, indexes the contained underlying array linearly.
   * Can lead to unexpected results in multidimensional tensors.
   */
  void set(ftype item, tensorDim_t idx) { values->set(item, idx); }

  void setDevice(const Device d) noexcept;
  Device getDevice() const noexcept { return values->getDevice(); }

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
  static void setDefaultDevice(const Device d) noexcept { tensorValues_t::setDefaultDevice(d); }
  static Device getDefaultDevice() noexcept { return tensorValues_t::getDefaultDevice(); }
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
    res.reset(0.0f);

    // tune in accordance with cache-line size
    constexpr tensorSize_t TILESIZE = MemoryLayout::CACHE_LINE_BYTES / sizeof(ftype) * 4;
    matmul::MatmulTile<ftype, TILESIZE, TILESIZE, TILESIZE> tiles;

    for(tensorSize_t i = 0; i < nRowsLeft; i += TILESIZE) {
      for(tensorSize_t j = 0; j < nColsRight; j += TILESIZE) {
        tiles.clearResult();

        for(tensorSize_t k0 = 0; k0 < nColsLeft; k0 += TILESIZE) {
          tiles.loadLeft(left.values->data(), i, k0, nRowsLeft, nColsLeft);
          tiles.loadRight(right.values->data(), k0, j, nRowsRight, nColsRight);

          for(tensorSize_t n = 0; n < TILESIZE; n++) { // rows left
          const tensorSize_t leftTileRowOffset = n * TILESIZE;

            for(tensorSize_t m = 0; m < TILESIZE; m++) { // cols left
              const tensorSize_t rightTileRowOffset = m * TILESIZE;

              const ftype leftVal = tiles.left[leftTileRowOffset + m];
              for(tensorSize_t kk = 0; kk < TILESIZE; kk++) {
                tiles.result[leftTileRowOffset + kk] += leftVal * tiles.right[rightTileRowOffset + kk];
              }
            }
          }
        }

        tiles.addResult(res.values->data() + resOffset, i, j, nRowsLeft, nColsRight);
      }
    }
  }
/*   else if constexpr (!transposeLeft && transposeRight) {
    for (tensorSize_t i = 0; i < nRowsLeft; i++) {
      const tensorSize_t leftOffset = i * nColsLeft;
      const tensorSize_t resOffset = i * nRowsRight;

      for(tensorSize_t j = 0; j < nRowsRight; j++) {
        const tensorSize_t rightOffset = j * nColsRight;

        ftype sum = 0.0f;
        for (tensorSize_t k = 0; k < nColsRight; k++) {
          sum += left.values->data()[leftOffset + k] * right.values->data()[rightOffset + k];
        }

        res.values->data()[resOffset + j] = sum;
      }
    }
  } */
/*   else if constexpr (transposeLeft && !transposeRight) {
    res.reset(0.0f);

    constexpr tensorSize_t TILESIZE = MemoryLayout::CACHE_LINE_BYTES / sizeof(ftype); // each tile fits neatly in cache-line
    matmul::MatmulTile<ftype, TILESIZE, TILESIZE, TILESIZE> tiles;

    for(tensorSize_t k0 = 0; k0 < nRowsLeft; k0 += TILESIZE) {
      for (tensorSize_t i = 0; i < nColsLeft; i += TILESIZE) {
        for(tensorSize_t j = 0; j < nColsRight; j += TILESIZE) {
          tiles.clearResult();

          for (tensorSize_t k0 = 0; k0 < nColsLeft; k0 += TILESIZE) {
            tiles.loadLeftTransposed(left.values->data(), i, k0, nRowsLeft, nColsLeft);
            tiles.loadRight(right.values->data(), k0, j, nRowsRight, nColsRight);

            for (tensorSize_t n = 0; n < TILESIZE; n++) { // rows left
            const tensorSize_t leftTileRowOffset = n * TILESIZE;

              for (tensorSize_t m = 0; m < TILESIZE; m++) { // cols left
                const tensorSize_t rightTileRowOffset = m * TILESIZE;

                const ftype leftVal = tiles.left[leftTileRowOffset + m];
                for (tensorSize_t kk = 0; kk < TILESIZE; kk++) {
                  //sum += tiles.left[leftTileOffset + k] * tiles.right[k * TILESIZE + n];
                  tiles.result[leftTileRowOffset + kk] += leftVal * tiles.right[rightTileRowOffset + kk];
                }
              }
            }
          }

          tiles.addResult(res.values->data() + resOffset, i, j, nRowsLeft, nColsRight);
        }
      }
    }
  } */
  /* else {
    for (tensorSize_t i = 0; i < nColsLeft; i++) {
      const tensorSize_t resOffset = i * nRowsRight;

      for (tensorSize_t j = 0; j < nRowsRight; j++) {
        const tensorSize_t rightOffset = j * nColsRight;

        ftype sum = 0.0f;
        for (tensorSize_t k = 0; k < nRowsLeft; k++) {
          sum += left.values->data()[k * nColsLeft + i] * right.values->data()[rightOffset + k];
        }

        res.values->data()[resOffset + j] = sum;
      }
    }
  } */
  else {
    const tensorSize_t M = transposeLeft ? nColsLeft : nRowsLeft;
    const tensorSize_t K = transposeLeft ? nRowsLeft : nColsLeft;
    const tensorSize_t N = transposeRight ? nRowsRight : nColsRight;

    for(tensorSize_t i = 0; i < M; i++) {
      for(tensorSize_t j = 0; j < N; j++) {
        ftype sum = 0;

        for(tensorSize_t k = 0; k < K; k++) {
          tensorSize_t leftIdx = transposeLeft ? leftOffset + k * nColsLeft + i
                                              : leftOffset + i * nColsLeft + k;

          tensorSize_t rightIdx = transposeRight ? rightOffset + j * nColsRight + k
                                                : rightOffset + k * nColsRight + j;

          sum += left.values->data()[leftIdx] * right.values->data()[rightIdx];
        }

        res.values->data()[resOffset + i * N + j] = sum;
      }
    }
  }
}

#if defined(USE_AVX)
static_assert(false, "TODO: implement AVX 1 version of matmul kernel");
#endif

/**
 * @brief Name says it all. Inplace operation on res.
 *
 * Transposition baked into kernel. Assumption is that kernels are not transposed, but we treat them
 * as if they were for increased speed without the actual transposition happening.
 */

#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX)
template<bool transposeLeft, bool transposeRight>
void Tensor::matMul2DCpuAvx(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset,
                               const tensorSize_t leftOffset, const tensorSize_t rightOffset) {

  // physical, not logically as in the transposition
  const auto nRowsLeft = static_cast<tensorSize_t>(left.dims.get(-2));
  const auto nColsLeft = static_cast<tensorSize_t>(left.dims.get(-1));
  const auto nColsRight = static_cast<tensorSize_t>(right.dims.get(-1));
  const auto nRowsRight = static_cast<tensorSize_t>(right.dims.get(-2));

  if constexpr (!transposeLeft && !transposeRight) {
    res.reset(0.0f);

    // tune in accordance with cache-line size
    constexpr tensorSize_t TILESIZE = MemoryLayout::CACHE_LINE_BYTES / sizeof(ftype) * 4;
    matmul::MatmulTile<ftype, TILESIZE, TILESIZE, TILESIZE> tiles;

    for(tensorSize_t i = 0; i < nRowsLeft; i += TILESIZE) {
      for(tensorSize_t j = 0; j < nColsRight; j += TILESIZE) {
        tiles.clearResult();

        for(tensorSize_t k0 = 0; k0 < nColsLeft; k0 += TILESIZE) {
          tiles.loadLeft(left.values->data(), i, k0, nRowsLeft, nColsLeft);
          tiles.loadRight(right.values->data(), k0, j, nRowsRight, nColsRight);

          for(tensorSize_t n = 0; n < TILESIZE; n++) { // rows left
          const tensorSize_t leftTileRowOffset = n * TILESIZE;

            for(tensorSize_t m = 0; m < TILESIZE; m++) { // cols left
              const tensorSize_t rightTileRowOffset = m * TILESIZE;

              const ftype leftVal = tiles.left[leftTileRowOffset + m];
              const __m256 leftValVec = _mm256_set1_ps(leftVal); // broadcasted load

              for(tensorSize_t kk = 0; kk < TILESIZE; kk += 8) {
                __m256 rightVec  = _mm256_load_ps(&tiles.right[rightTileRowOffset + kk]);
                __m256 resultVec = _mm256_load_ps(&tiles.result[leftTileRowOffset + kk]);

                resultVec = _mm256_fmadd_ps(leftValVec, rightVec, resultVec);
                _mm256_store_ps(&tiles.result[leftTileRowOffset + kk], resultVec);
              }
            }
          }
        }

        tiles.addResult(res.values->data() + resOffset, i, j, nRowsLeft, nColsRight);
      }
    }
  }
  else {
    const tensorSize_t M = transposeLeft ? nColsLeft : nRowsLeft;
    const tensorSize_t K = transposeLeft ? nRowsLeft : nColsLeft;
    const tensorSize_t N = transposeRight ? nRowsRight : nColsRight;

    for(tensorSize_t i = 0; i < M; i++) {
      for(tensorSize_t j = 0; j < N; j++) {
        ftype sum = 0;

        for (tensorSize_t k = 0; k < K; k++) {
          tensorSize_t leftIdx = transposeLeft ? leftOffset + k * nColsLeft + i
                                              : leftOffset + i * nColsLeft + k;

          tensorSize_t rightIdx = transposeRight ? rightOffset + j * nColsRight + k
                                                : rightOffset + k * nColsRight + j;

          sum += left.values->data()[leftIdx] * right.values->data()[rightIdx];
        }

        res.values->data()[resOffset + i * N + j] = sum;
      }
    }
  }
}
#endif // defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX)