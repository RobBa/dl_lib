/**
 * @file matmul_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "graph_node.h"

#include <memory>

namespace graph {
  class MatMulNode final : public GraphNode {
    public:
      explicit MatMulNode(Tensor* t1, Tensor* t2): GraphNode({t1, t2}) {}

      MatMulNode(const MatMulNode& other) = delete;
      MatMulNode& operator=(const MatMulNode& other) = delete;

      MatMulNode(MatMulNode&& other) = default;
      MatMulNode& operator=(MatMulNode&& other) = default;

      ~MatMulNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
