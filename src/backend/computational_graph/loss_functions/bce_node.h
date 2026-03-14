/**
 * @file bce_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"
#include "utility/global_params.h"

namespace cgraph {
  class BceNode final : public GraphNode {
    private:
      const std::shared_ptr<const Tensor> yTrue;
      const ftype bSize;

    public:
      explicit BceNode(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> yPred) 
        : GraphNode({yPred}), yTrue{std::move(y)}, bSize{static_cast<ftype>(yPred->getDims()[0])}
        {
          assert(yPred->getDims()==yTrue->getDims());

          if(!yPred->getRequiresGrad()){
            std::__throw_invalid_argument("yPred must be a graph node");
          }
        }

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}