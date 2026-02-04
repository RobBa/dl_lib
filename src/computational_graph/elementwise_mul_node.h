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
      ElementwiseMulNode(Tensor* t1, Tensor* t2) {
        parents = {t1, t2};
      }

      ElementwiseMulNode(const ElementwiseMulNode& other) = delete;
      ElementwiseMulNode& operator=(const ElementwiseMulNode& other) = delete;

      ElementwiseMulNode(ElementwiseMulNode&& other) = delete;
      ElementwiseMulNode& operator=(ElementwiseMulNode&& other) = delete;

      ~ElementwiseMulNode() noexcept = default; 

      std::vector<Tensor> backward(const Tensor& upstream_grad) override;
  };
}
