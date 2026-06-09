/**
 * @file initializers.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-06-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif // __CUDA

#include "shared/global_params.h"

namespace cuda_impl {
  void scaleArr(ftype* const data, ftype scale, ftype shift, tensorSize_t size);
}