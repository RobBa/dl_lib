/**
 * @file scalarmul_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "graph_node.h"

namespace graph {
  class ScalarMulNode final : public GraphNode {
    public:
      ScalarMulNode(Tensor* t1, Tensor* t2) {
        parents = {t1, t2};
      }

      ScalarMulNode(const ScalarMulNode& other) = delete;
      ScalarMulNode& operator=(const ScalarMulNode& other) = delete;

      ScalarMulNode(ScalarMulNode&& other) = default;
      ScalarMulNode& operator=(ScalarMulNode&& other) = default;

      ~ScalarMulNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
