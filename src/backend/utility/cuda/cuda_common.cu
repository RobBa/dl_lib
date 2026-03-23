/**
 * @file cuda_common.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be included without CUDA enabled");
#endif

#include "cuda_common.cuh"

#include <iostream>

void utility::gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
  {
      //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line;
      if (abort) exit(code);
  }
}