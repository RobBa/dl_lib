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

#include "computational_graph/graph_node.h"
#include "utility/utils.h"

#include <memory>

namespace cgraph {
  class DLLIB_API MatMulNode final : public GraphNode {
    private:
      const bool transposeLeft;
      const bool transposeRight;

    public:
      explicit MatMulNode(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, 
                          const bool transposeLeft, const bool transposeRight) 
        : GraphNode({std::move(t1), std::move(t2)}), transposeLeft{transposeLeft}, transposeRight{transposeRight} {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
