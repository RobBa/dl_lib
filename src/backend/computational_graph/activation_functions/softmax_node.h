/**
 * @file softmax_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-15
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"
#include "utility/global_params.h"

namespace cgraph {
  class SoftmaxNode final : public GraphNode {
    private:
      const std::shared_ptr<const Tensor> softmax;

    public:
      explicit SoftmaxNode(std::shared_ptr<Tensor> t, std::shared_ptr<const Tensor> softmax) 
        : GraphNode({std::move(t)}), softmax{std::move(softmax)}
        {
          assert(softmax->getSize()==parents[0]->getDims()[0]);
        }

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}