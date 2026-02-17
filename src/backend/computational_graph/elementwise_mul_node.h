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

#include "graph_node.h"

namespace graph {
  class ElementwiseMulNode final : public GraphNode {
    public:
      explicit ElementwiseMulNode(Tensor* t1, Tensor* t2) : GraphNode({t1, t2}) {}

      ElementwiseMulNode(const ElementwiseMulNode& other) = delete;
      ElementwiseMulNode& operator=(const ElementwiseMulNode& other) = delete;

      ElementwiseMulNode(ElementwiseMulNode&& other) = default;
      ElementwiseMulNode& operator=(ElementwiseMulNode&& other) = default;

      ~ElementwiseMulNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
