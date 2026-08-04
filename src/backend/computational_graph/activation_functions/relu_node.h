/**
 * @file relu_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-15
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"
#include "utility/utils.h"

#include <memory>

namespace cgraph {
  class DLLIB_API ReLuNode final : public GraphNode {
    public:
      explicit ReLuNode(std::shared_ptr<Tensor> t) 
        : GraphNode({std::move(t)}) {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
