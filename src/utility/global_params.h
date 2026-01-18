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

using ftype = float; // TODO: make compiler flag

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
constexpr int MAX_TENSOR_DIMS = 4;
using tensorDim_t = std::uint16_t;
using tensorSize_t = std::uint32_t;