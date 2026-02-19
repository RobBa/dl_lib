/**
 * @file getter_node.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-18
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "graph_node.h"

namespace graph{
  /**
   * @brief When calling a get function, say as in 
   * loss += myTensor[i], then we need to build a graph in between 
   * the myTensor[i] and the myTensor object. Hence this node.
   * 
   */
  class GetterNode final : public GraphNode {
    public:
      explicit GetterNode(std::shared_ptr<Tensor> t) 
        : GraphNode({std::move(t)}) {}

      GetterNode(const GetterNode& other) = delete;
      GetterNode& operator=(const GetterNode& other) = delete;

      GetterNode(GetterNode&& other) = default;
      GetterNode& operator=(GetterNode&& other) = default;

      ~GetterNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };}
