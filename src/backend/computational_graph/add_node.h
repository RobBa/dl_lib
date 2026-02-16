/**
 * @file add_node.h
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

namespace graph {
  class AddNode final : public GraphNode {  
    public:
      AddNode(Tensor* t1, Tensor* t2) {
        parents = {t1, t2};
      }

      AddNode(const AddNode& other) = delete;
      AddNode& operator=(const AddNode& other) = delete;

      AddNode(AddNode&& other) = default;
      AddNode& operator=(AddNode&& other) = default;

      ~AddNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}