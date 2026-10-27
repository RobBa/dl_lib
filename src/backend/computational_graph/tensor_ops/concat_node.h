/**
 * @file concat_node.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"
#include "utility/utils.h"

#include <memory>

namespace cgraph {
  class DLLIB_API ConcatNode final : public GraphNode {
    private:
      const tensorDim_t dim;

    public:
      explicit ConcatNode(std::vector<std::shared_ptr<Tensor>> parents, const tensorDim_t dim) 
        : GraphNode(std::move(parents)), dim{dim} {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
