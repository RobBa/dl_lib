/**
 * @file rmsprop.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "optimizer_base.h"
#include "utility/global_params.h"

#include <unordered_map>

namespace train {
  class RmsPropOptimizer final : public OptimizerBase {
    private:
      const ftype decay;
      std::unordered_map<Tensor*, std::unique_ptr<Tensor>> movingAvg;

    public:
        RmsPropOptimizer(std::vector< std::shared_ptr<Tensor> > params, ftype lr, ftype decay) 
          : OptimizerBase(std::move(params), lr), decay{decay} 
          {
            for(const auto& param: params) {
              movingAvg[param.get()] = nullptr; // lazy initialization
            }
          }

        void step() override;
  };
}