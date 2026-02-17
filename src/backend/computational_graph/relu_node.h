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

#include "graph_node.h"

#include <memory>

namespace graph {
  class ReLuNode final : public GraphNode {
    public:
      explicit ReLuNode(std::shared_ptr<Tensor> t) 
        : GraphNode({std::move(t)}) {}

      ReLuNode(const ReLuNode& other) = delete;
      ReLuNode& operator=(const ReLuNode& other) = delete;

      ReLuNode(ReLuNode&& other) = default;
      ReLuNode& operator=(ReLuNode&& other) = default;

      ~ReLuNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
