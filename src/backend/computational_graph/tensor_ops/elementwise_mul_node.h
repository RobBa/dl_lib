/**
 * @file elementwise_mul_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-04
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"

namespace graph {
  class ElementwiseMulNode final : public GraphNode {
    public:
      explicit ElementwiseMulNode(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) 
        : GraphNode({std::move(t1), std::move(t2)}) {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
