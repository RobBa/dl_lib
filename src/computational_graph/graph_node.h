/**
 * @file igraph_node.h
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
#include <memory>

namespace graph {
  class GraphNode {
    friend class TopologicalSort;
    
    protected:
      std::vector< std::shared_ptr<Tensor> > parents;

    public:
      virtual std::vector<Tensor> backward(const Tensor& upstream_grad) = 0;
  };
}
