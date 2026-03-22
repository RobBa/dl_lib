/**
 * @file cuda_common.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifdef __CUDA

#include "cuda.h"

namespace utility {  
  void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
}

#define gpuErrchk(ans) { utility::gpuAssert((ans), __FILE__, __LINE__); }

#endif // __CUDA