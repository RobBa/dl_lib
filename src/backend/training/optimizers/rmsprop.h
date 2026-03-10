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
    private:
      void step(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) override;
      
    public:
        RmsPropOptimizer(std::vector< std::shared_ptr<Tensor> >& params, 
            std::shared_ptr<LossBase> loss, ftype lr, size_t epochs, tensorDim_t bsize) 
          : OptimizerBase(params, loss, lr, epochs, bsize) { }

        // TODO: print
  };
}