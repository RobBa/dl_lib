/**
 * @file main.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-05-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __CUDA
static_assert(false, "File should not be compiled without CUDA enabled");
#endif // __CUDA

#include <gtest/gtest.h>

#include "system/sys_functions.h"

class CudaEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    // cuda warmup to avoid context initialization costs
    void* tmp;
    cudaMalloc(&tmp, 1);
    cudaFree(tmp);
    cudaDeviceSynchronize();
  }
};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CudaEnvironment());
  sys::setRandomSeed(42);
  return RUN_ALL_TESTS();
}