/**
 * @file global_params.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <cstdint>

#include <array>

using ftype = float; // TODO: make compiler flag?

/** 
 * IMPORTANT: For the following block we assume that
 * no overflows can happen, therefore this part of the 
 * code must be thoroughly checked. For example, if you 
 * intend to create 4-dimensional tensors, then the maximum
 * size of the tensor maxSize = dim1 * dim2 * dim3 * dim4,
 * dim1...dim4 are of type tensorDim_t, 
 * must fit into the datatype tensorSize_t. We DO NOT CHECK
 * FOR OVERFLOWS. Similarly, we assume that all dimensions you
 * request fit into datatype tensorDim_t.
 */ 
using tensorDim_t = std::uint32_t;
using tensorSize_t = std::uint32_t;

/**
 * Note: If we want to narrow down tensorDim_t, say for embedded
 * devices, we can easily run into subtle bugs, as overflows of 
 * integers are not detected. A quick way out could be the 
 * following struct, which is currently not implemented yet.
 * 
 * struct tensorDim_t {
 *   uint16_t value;
 *   tensorSize_t operator*(const tensorDim_t& other) const {
 *     return static_cast<tensorSize_t>(value) * other.value;
 *   }
 * };
 */

constexpr tensorDim_t MAX_NDIMS = 6;

// we assert this here so during conversions of tensorDim_t to 
// tensorSize_t we do not need to cast explicitely
static_assert(sizeof(tensorDim_t)<=sizeof(tensorSize_t));

// ----------------- Numerical stability -------------------

constexpr ftype EPS_CROSSENTROPY = 1e-5;
constexpr ftype EPS_BCE = 1e-5;
constexpr ftype EPS_RMSE = 1e-9;

// ----------------- Default values ------------------------

constexpr ftype EPS_LEAKY_RELU = 0.01;
