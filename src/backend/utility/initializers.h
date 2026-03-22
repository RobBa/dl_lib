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

namespace utility {
  class InitializerBase {
    protected: 
      static inline std::optional<unsigned int> randomSeed_opt = std::nullopt;

    public:
      InitializerBase() = default;

      virtual ~InitializerBase() = default;
      virtual ftype drawNumber() const = 0;

      static void setSeed(unsigned int s) noexcept { randomSeed_opt = s; }
  };

  class GaussianInitializer final : public InitializerBase {
    private:
      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::normal_distribution<ftype> dist;

    public:
        GaussianInitializer(ftype stddev) : gen{rd()}, dist{0, stddev} 
        {
          if(randomSeed_opt){
            gen = std::mt19937{randomSeed_opt.value()};
          }
        }

        GaussianInitializer(ftype stddev, unsigned int seed) 
          : dist{0, stddev}
        {
          gen = std::mt19937{seed};
        }
        
        ftype drawNumber() const override;
  };

  class UniformXavierInitializer final : public InitializerBase {
    private:
      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::uniform_real_distribution<ftype> dist;

      ftype computeRange(ftype nInputs, ftype nOutputs);

    public:
        UniformXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs)
          : gen{rd()}, dist{-computeRange(nInputs, nOutputs), computeRange(nInputs, nOutputs)} 
          {
            if(randomSeed_opt){
              gen = std::mt19937{randomSeed_opt.value()};
            }
          }

        UniformXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs, unsigned int seed)
          : dist{-computeRange(nInputs, nOutputs), computeRange(nInputs, nOutputs)}
          {
            gen = std::mt19937{seed};
          }

        ftype drawNumber() const override;
  };

  class NormalXavierInitializer final : public InitializerBase {
    private:
      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::normal_distribution<ftype> dist;

      ftype computeSigma(ftype nInputs, ftype nOutputs);

    public:
        NormalXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs)
          : gen{rd()}, dist{0, computeSigma(nInputs, nOutputs)}
          {
            if(randomSeed_opt){
              gen = std::mt19937{randomSeed_opt.value()};
            }
          }

        NormalXavierInitializer(tensorDim_t nInputs, tensorDim_t nOutputs, unsigned int seed)
          : dist{0, computeSigma(nInputs, nOutputs)}
          {
            gen = std::mt19937{seed};
          }

        ftype drawNumber() const override;
  };
}
