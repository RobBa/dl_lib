/**
 * @file matmul_tile.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "shared/global_params.h"
#include "utility/memory_layout.h"

#include "utility/utils.h"

#include <array>
#include <type_traits>

namespace matmul {
  /**
   * @brief Used for matrix multiplication. Three tiles that can be accessed and transposed easily.
   * 
   * Expected shapes and template params: Left = (MxK), right = (KxN), res = (MxN). 
   * We keep the shapes as template parameters to enable optimizations on them.
   * 
   * @tparam T The datatype of the computation.
   */
  template<typename T, tensorSize_t TileM, tensorSize_t TileK, tensorSize_t TileN>
  requires std::is_floating_point_v<T>
  struct MatmulTile {
      alignas(MemoryLayout::CPU_TENSOR_ALIGNMENT) std::array<T, TileM * TileK> left{};
      alignas(MemoryLayout::CPU_TENSOR_ALIGNMENT) std::array<T, TileK * TileN> right{};
      alignas(MemoryLayout::CPU_TENSOR_ALIGNMENT) std::array<T, TileM * TileN> result{};

      void loadLeft(const T* const src, tensorSize_t row0, tensorSize_t col0, tensorSize_t nRows, tensorSize_t nCols);
      void loadRight(const T* const src, tensorSize_t row0, tensorSize_t col0, tensorSize_t nRows, tensorSize_t nCols);
      
      //void loadLeftTransposed(const T* const src, tensorSize_t row0, tensorSize_t col0, tensorSize_t nRows, tensorSize_t nCols);
      void loadRightTransposed(const T* const src, tensorSize_t srcStride, tensorSize_t row0, tensorSize_t col0);
      
      // TODO: this would be an optimization
      //void storeResult(T* const dst, tensorSize_t row0, tensorSize_t col0, tensorSize_t nRows, tensorSize_t nCols);
      void clearResult() noexcept {
        std::fill(result.begin(), result.begin() + TileM * TileN, T{0.0f});
      }
      void addResult(T* const dst, tensorSize_t row0, tensorSize_t col0, tensorSize_t nRows, tensorSize_t nCols);
  };
}

/**
 * @brief Loads left from a tensor (src).
 * 
 * @param row0 Row to start from.
 * @param col0 Col to start from.
 * @param nRows Number of rows of src.
 * @param nCols Number of columns of src.
 */
template<typename T, tensorSize_t TileM, tensorSize_t TileK, tensorSize_t TileN>
requires std::is_floating_point_v<T>
void matmul::MatmulTile<T, TileM, TileK, TileN>::loadLeft(const T* const src,
                                                          const tensorSize_t row0, const tensorSize_t col0, 
                                                          const tensorSize_t nRows, const tensorSize_t nCols) {
  const tensorSize_t maxIdxRows = std::min(row0 + TileM, nRows);
  const tensorSize_t maxIdxCols = std::min(col0 + TileK, nCols);

  const tensorSize_t validRows = maxIdxRows - row0;
  const tensorSize_t validCols = maxIdxCols - col0;
  
  for(tensorSize_t row = row0; row < maxIdxRows; row++) {
    const tensorSize_t srcRowOffset = nCols * row;
    const tensorSize_t tileRowOffset = (row - row0) * TileK;
    
    tensorSize_t tileCol = 0;
    for(tensorSize_t col = col0; col < maxIdxCols; col++) {
      left[tileRowOffset + tileCol] = src[srcRowOffset + col];
      tileCol++;
    }

    if (validCols < TileK) {
      std::fill(left.begin() + tileRowOffset + validCols,
                left.begin() + tileRowOffset + TileK,
                T{0.0f});
    }
  }

  if (validRows < TileM) {
    std::fill(left.begin() + validRows * TileK, left.end(), T{0.0f});
  }
}

/**
 * @brief Loads right tensor from a tensor (src).
 * 
 * @param row0 Row to start from.
 * @param col0 Col to start from.
 * @param nRows Number of rows of src.
 * @param nCols Number of columns of src.
 */
template<typename T, tensorSize_t TileM, tensorSize_t TileK, tensorSize_t TileN>
requires std::is_floating_point_v<T>
void matmul::MatmulTile<T, TileM, TileK, TileN>::loadRight(const T* const src,
                                                           const tensorSize_t row0, const tensorSize_t col0, 
                                                           const tensorSize_t nRows, const tensorSize_t nCols) {
  const tensorSize_t maxIdxRows = std::min(row0 + TileK, nRows);
  const tensorSize_t maxIdxCols = std::min(col0 + TileN, nCols);

  const tensorSize_t validRows = maxIdxRows - row0;
  const tensorSize_t validCols = maxIdxCols - col0;

  for (tensorSize_t row = row0; row < maxIdxRows; row++) {
    const tensorSize_t srcRowOffset = nCols * row;
    const tensorSize_t tileRowOffset = (row - row0) * TileN;

    tensorSize_t tileCol = 0;
    for (tensorSize_t col = col0; col < maxIdxCols; col++) {
      right[tileRowOffset + tileCol] = src[srcRowOffset + col];
      tileCol++;
    }

    if(validCols < TileN) {
      std::fill(right.begin() + tileRowOffset + validCols,
                right.begin() + tileRowOffset + TileN,
                T{0.0f});
    }
  }

  if(validRows < TileK) {
    std::fill(right.begin() + validRows * TileN, right.end(), T{0.0f});
  }
}

/**
 * @brief Like loadLeft, but transposes the matrix when loading into tile.
 * 
 * @param row0 Row to start from.
 * @param col0 Col to start from.
 * @param nRows Number of rows of src.
 * @param nCols Number of columns of src.
 */
/* template<typename T, tensorSize_t TileM, tensorSize_t TileK, tensorSize_t TileN>
requires std::is_floating_point_v<T>
void matmul::MatmulTile<T, TileM, TileK, TileN>::loadLeftTransposed(const T* const src,
                                                                    const tensorSize_t row0, const tensorSize_t col0, 
                                                                    const tensorSize_t nRows, const tensorSize_t nCols) {
  const tensorSize_t maxIdxRows = std::min(row0 + TileM, nRows);
  const tensorSize_t maxIdxCols = std::min(col0 + TileK, nCols);

  const tensorSize_t validRows = maxIdxRows - row0;
  const tensorSize_t validCols = maxIdxCols - col0;
  
  for(tensorSize_t row = row0; row < maxIdxRows; row++) {
    const tensorSize_t srcRowOffset = nCols * row;
    const tensorSize_t tileRowOffset = (row - row0) * TileK;
    
    tensorSize_t tileCol = 0;
    for(tensorSize_t col = col0; col < maxIdxCols; col++) {
      left[tileRowOffset + tileCol] = src[srcRowOffset + col];
      tileCol++;
    }

    if (validCols < TileK) {
      std::fill(left.begin() + tileRowOffset + validCols,
                left.begin() + tileRowOffset + TileK,
                T{0.0f});
    }
  }

  if (validRows < TileM) {
    std::fill(left.begin() + validRows * TileK, left.end(), T{0.0f});
  }
} */

/**
 * @brief Like loadRight, but transposes the matrix when loading into tile.
 * 
 * @param row0 Row to start from.
 * @param col0 Col to start from.
 * @param nRows Number of rows of src.
 * @param nCols Number of columns of src.
 */
template<typename T, tensorSize_t TileM, tensorSize_t TileK, tensorSize_t TileN>
requires std::is_floating_point_v<T>
void matmul::MatmulTile<T, TileM, TileK, TileN>::loadRightTransposed(const T* const src,
                                                           const tensorSize_t row0, const tensorSize_t col0, 
                                                           const tensorSize_t nRows, const tensorSize_t nCols) {
  const tensorSize_t maxIdxRows = std::min(row0 + TileK, nRows);
  const tensorSize_t maxIdxCols = std::min(col0 + TileN, nCols);

  const tensorSize_t validRows = maxIdxRows - row0;
  const tensorSize_t validCols = maxIdxCols - col0;

  for (tensorSize_t row = row0; row < maxIdxRows; row++) {
    const tensorSize_t srcRowOffset = nCols * row;
    const tensorSize_t tileRowOffset = (row - row0) * TileN;

    tensorSize_t tileCol = 0;
    for (tensorSize_t col = col0; col < maxIdxCols; col++) {
      right[tileRowOffset + tileCol] = src[srcRowOffset + col];
      tileCol++;
    }

    if(validCols < TileN) {
      std::fill(right.begin() + tileRowOffset + validCols,
                right.begin() + tileRowOffset + TileN,
                T{0.0f});
    }
  }

  if(validRows < TileK) {
    std::fill(right.begin() + validRows * TileN, right.end(), T{0.0f});
  }
}

/**
 * @brief Store back into tensor (dst) from tile result. Does an add-operation.
 * 
 * @param row0 Row to start writing from.
 * @param col0 Column to start writing from.
 * @param nRows Number of rows of dst.
 * @param nCols Number of columns of dst.
 */
template<typename T, tensorSize_t TileM, tensorSize_t TileK, tensorSize_t TileN>
requires std::is_floating_point_v<T>
void matmul::MatmulTile<T, TileM, TileK, TileN>::addResult(T* const dst, 
                                                                  tensorSize_t row0, tensorSize_t col0, 
                                                                  tensorSize_t nRows, tensorSize_t nCols) {
  const tensorSize_t maxIdxRows = std::min(row0 + TileM, nRows);
  const tensorSize_t maxIdxCols = std::min(col0 + TileN, nCols);

  for (tensorSize_t row = row0; row < maxIdxRows; row++) {
    const tensorSize_t dstRowOffset = nCols * row;
    const tensorSize_t tileRowOffset = (row - row0) * TileN;
    
    tensorSize_t tileCol = 0;
    for (tensorSize_t col = col0; col < maxIdxCols; col++) {
      dst[dstRowOffset + col] += result[tileRowOffset + tileCol];
      tileCol++;
    }
  }
}
