/**
 * @file initializers.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "global_params.h"

#include <random>
#include <algorithm>

#include <memory>
#include <optional>

#ifdef __CUDA
#include <curand.h>
#endif

namespace utility {
  class InitializerBase {
    protected: 
      static inline std::optional<unsigned int> randomSeed_opt = std::nullopt;

      virtual ftype drawNumber() const = 0;

    #ifdef __CUDA
      curandGenerator_t cuGen;
    #endif

    public:
      InitializerBase() {
      #ifdef __CUDA
        curandCreateGenerator(&cuGen, CURAND_RNG_PSEUDO_DEFAULT);
        if(randomSeed_opt) {
          curandSetPseudoRandomGeneratorSeed(cuGen, randomSeed_opt.value());
        }
      #endif
      }

      InitializerBase(unsigned int seed) {
      #ifdef __CUDA
        curandCreateGenerator(&cuGen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(cuGen, seed);
      #endif
      }

      virtual ~InitializerBase() {
      #ifdef __CUDA
        curandDestroyGenerator(cuGen);
      #endif
      }

      static void setSeed(unsigned int s) noexcept { 
        randomSeed_opt = s; 
      }

      void fillRange(ftype* const data, tensorSize_t size) const;
      virtual void fillRangeGpu(float* const data, tensorSize_t size) const = 0;
      virtual void fillRangeGpu(double* const data, tensorSize_t size) const = 0;
  };

  class GaussianInitializer final : public InitializerBase {
    private:
      const ftype stddev;

      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::normal_distribution<ftype> dist;

      ftype drawNumber() const override;

    public:
        GaussianInitializer(ftype stddev) 
          : stddev{stddev}, gen{rd()}, dist{0.0f, stddev} 
        {
          if(randomSeed_opt) {
            gen = std::mt19937{randomSeed_opt.value()};
          }
        }

        GaussianInitializer(ftype stddev, unsigned int seed) 
          : InitializerBase(seed), stddev{stddev}, dist{0.0f, stddev}
        {
          gen = std::mt19937{seed};
        }
        
        void fillRangeGpu(float* const data, tensorSize_t size) const override;
        void fillRangeGpu(double* const data, tensorSize_t size) const override;
  };

  class UniformXavierInitializer final : public InitializerBase {
    private:
      const ftype range;

      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::uniform_real_distribution<ftype> dist;

      ftype computeRange(ftype nInputs, ftype nOutputs);
      ftype drawNumber() const override;

    public:
        UniformXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs)
          : range{computeRange(nInputs, nOutputs)}, gen{rd()}, dist{-range, range} 
          {
            if(randomSeed_opt) {
              gen = std::mt19937{randomSeed_opt.value()};
            }
          }

        UniformXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs, unsigned int seed)
          : InitializerBase(seed), range{computeRange(nInputs, nOutputs)}, gen{rd()}, dist{-range, range} 
          {
            gen = std::mt19937{seed};
          }

        void fillRangeGpu(float* const data, tensorSize_t size) const override;
        void fillRangeGpu(double* const data, tensorSize_t size) const override;  
  };

  class NormalXavierInitializer final : public InitializerBase {
    private:
      const ftype sigma;

      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::normal_distribution<ftype> dist;

      ftype computeSigma(ftype nInputs, ftype nOutputs);
      ftype drawNumber() const override;

    public:
        NormalXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs)
          : sigma{computeSigma(nInputs, nOutputs)}, gen{rd()}, dist{0, sigma}
          {
            if(randomSeed_opt) {
              gen = std::mt19937{randomSeed_opt.value()};
            }
          }

        NormalXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs, unsigned int seed)
          : InitializerBase(seed), sigma{computeSigma(nInputs, nOutputs)}, gen{rd()}, dist{0, sigma}
          {
            gen = std::mt19937{seed};
          }

        void fillRangeGpu(float* const data, tensorSize_t size) const override;
        void fillRangeGpu(double* const data, tensorSize_t size) const override;  
  };
}
