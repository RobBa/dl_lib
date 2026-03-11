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

namespace train {
  class RmsPropOptimizer final : public OptimizerBase {
    public:
        RmsPropOptimizer(std::vector< std::shared_ptr<Tensor> > params, ftype lr) 
          : OptimizerBase(std::move(params), lr) { }

        void step() override;
  };
}