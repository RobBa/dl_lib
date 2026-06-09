/**
 * @file initializers.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "initializers.h"

#include <cmath>
#include <stdexcept>

#ifdef __CUDA
#include "utility/utils.h"

#include "utility/cuda/cuda_common.cuh"
#include "shared/cuda/initializers.cuh"

#include <type_traits>
#include <cassert>
#endif

using namespace std;
using namespace utility;

void InitializerBase::fillRange(ftype* const data, tensorSize_t size) const {
  for(tensorSize_t i = 0; i < size; i++){
    data[i] = drawNumber();
  }
}

ftype GaussianInitializer::drawNumber() const {
  return dist(gen);
}

void GaussianInitializer::fillRangeGpu(float* const data, tensorSize_t size) const {
  #ifndef NDEBUG
  if(!is_same_v<ftype, float>)
    __throw_runtime_error("Should not be called when ftype!=float");
  #endif

  #ifdef __CUDA
    assert(size % 2 == 0 && "curandGenerateNormal requires even count");
    cuRandErrchk(curandGenerateNormal(cuGen, data, size, 0.0f, stddev));
  #else
    __throw_runtime_error("Not compiled with CUDA");
  #endif
}

void GaussianInitializer::fillRangeGpu(double* const data, tensorSize_t size) const {
  #ifndef NDEBUG
  if(!is_same_v<ftype, double>)
    __throw_runtime_error("Should not be called when ftype!=double");
  #endif

  #ifdef __CUDA
    assert(size % 2 == 0 && "curandGenerateNormal requires even count");
    cuRandErrchk(curandGenerateNormalDouble(cuGen, data, size, 0.0, stddev));
  #else
    __throw_runtime_error("Not compiled with CUDA");
  #endif
}

ftype UniformXavierInitializer::computeRange(ftype nInputs, ftype nOutputs) {
  return sqrt(6 / (nInputs + nOutputs));
}

ftype UniformXavierInitializer::drawNumber() const {
  return dist(gen);
}

void UniformXavierInitializer::fillRangeGpu(float* const data, tensorSize_t size) const {
  #ifndef NDEBUG
  if(!is_same_v<ftype, float>)
    __throw_runtime_error("Should not be called when ftype!=float");
  #endif

  #ifdef __CUDA
    cuRandErrchk(curandGenerateUniform(cuGen, data, size));
    // reinterpret cast to prevent compiler errors
    cuda_impl::scaleArr(reinterpret_cast<ftype* const>(data), 2.0f * range, range, size);
  #else
    __throw_runtime_error("Not compiled with CUDA");
  #endif
}

void UniformXavierInitializer::fillRangeGpu(double* const data, tensorSize_t size) const {
  #ifndef NDEBUG
  if(!is_same_v<ftype, double>)
    __throw_runtime_error("Should not be called when ftype!=double");
  #endif

  #ifdef __CUDA
    cuRandErrchk(curandGenerateUniformDouble(cuGen, data, size));
    // reinterpret cast to prevent compiler errors
    cuda_impl::scaleArr(reinterpret_cast<ftype* const>(data), 2.0 * range, range, size);
  #else
    __throw_runtime_error("Not compiled with CUDA");
  #endif
}

ftype NormalXavierInitializer::computeSigma(ftype nInputs, ftype nOutputs) {
  return sqrt(2/ (nInputs + nOutputs));
}

ftype NormalXavierInitializer::drawNumber() const {
  return dist(gen);
}

void NormalXavierInitializer::fillRangeGpu(float* const data, tensorSize_t size) const {
  #ifndef NDEBUG
  if(!is_same_v<ftype, float>)
    __throw_runtime_error("Should not be called when ftype!=float");
  #endif

  #ifdef __CUDA
    assert(size % 2 == 0 && "curandGenerateNormal requires even count");
    cuRandErrchk(curandGenerateNormal(cuGen, data, size, 0.0f, sigma));
  #else
    __throw_runtime_error("Not compiled with CUDA");
  #endif
}

void NormalXavierInitializer::fillRangeGpu(double* const data, tensorSize_t size) const {
  #ifndef NDEBUG
  if(!is_same_v<ftype, double>)
    __throw_runtime_error("Should not be called when ftype!=double");
  #endif

  #ifdef __CUDA
    assert(size % 2 == 0 && "curandGenerateNormal requires even count");
    cuRandErrchk(curandGenerateNormalDouble(cuGen, data, size, 0.0, sigma));
  #else
    __throw_runtime_error("Not compiled with CUDA");
  #endif
}