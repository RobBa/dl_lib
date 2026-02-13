/**
 * @file graph_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "topological_sort.h"

#include <vector>

namespace graph {
  class GraphNode {
    protected:
      std::vector<Tensor*> parents;

    public:
      virtual std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) = 0;
      
      const auto& getParents() const noexcept {
        return parents;
      }
  };
}
