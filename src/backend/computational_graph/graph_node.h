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

#include "data_modeling/tensor.h"

#include <vector>
#include <memory>

#include <utility>

namespace graph {
  class GraphNode {
    protected:
      std::vector< std::shared_ptr<Tensor> > parents;
      explicit GraphNode(std::vector< std::shared_ptr<Tensor> > parents) : parents{std::move(parents)}{}
      
    public:
      GraphNode(const GraphNode& other) = delete;
      GraphNode& operator=(const GraphNode& other) = delete;

      GraphNode(GraphNode&& other) = default;
      GraphNode& operator=(GraphNode&& other) = default;

      virtual ~GraphNode() noexcept = default; 

      virtual std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) = 0;
      
      const auto& getParents() const noexcept {
        return parents;
      }
  };
}
