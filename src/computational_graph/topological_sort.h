/**
 * @file topological_sort.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>

class Tensor; // to break circular dependency

namespace graph {
  /**
   * @brief Topological sort class. 
   * 
   * Kahn's algorithm, except that this one expects only one single starting node.
   * We use it for the backpropagation algorithm.
   */
  class TopologicalSort final {
    public:
      TopologicalSort() = delete;
      static std::vector< Tensor* > reverseSort(Tensor* root);
  };
}