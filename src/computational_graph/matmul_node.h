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
      MatMulNode(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
        parents = {t1, t2};
      }

      std::vector<Tensor> backward(const Tensor& upstream_grad) override;
  };
}
