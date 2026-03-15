/**
 * @file add_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"
#include "utility/global_params.h"

namespace cgraph {
  class CrossEntropyNode final : public GraphNode {
    private:
      const std::shared_ptr<const Tensor> yTrue;

    public:

      /**
       * @brief Expexted shapes are same as for CrossEntropyLoss.
       * 
       * @param y shape (batchsize)
       * @param yPred shape (batchsize, nclasses)
       */
      explicit CrossEntropyNode(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> yPred) 
        : GraphNode({std::move(yPred)}), yTrue{std::move(y)}
        {
          assert(parents[0]->getDims()==yTrue->getDims());
          if(!parents[0]->getRequiresGrad()){
            std::__throw_invalid_argument("yPred must be a graph node");
          }
        }

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}