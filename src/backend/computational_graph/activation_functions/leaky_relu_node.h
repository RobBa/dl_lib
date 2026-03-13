/**
 * @file leaky_relu_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"

#include <memory>

namespace cgraph {
  class LeakyReLuNode final : public GraphNode {
    private:
      const ftype eps;

    public:
      explicit LeakyReLuNode(std::shared_ptr<Tensor> t, const ftype eps) 
        : GraphNode({std::move(t)}), eps{eps} {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
