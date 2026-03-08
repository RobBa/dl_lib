/**
 * @file sgd.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "optimizer_base.h"

#include "utility/global_params.h"

namespace train {
  class SgdOptimizer final : public OptimizerBase {
    private:
      const ftype lr;
      
    public:
        SgdOptimizer(ftype lr) : lr{lr} 
        { }

        void step() override;

        // TODO: print
  };
}