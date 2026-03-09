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

#include "computational_graph/graph_node.h"

#include <vector>
#include <variant>

namespace graph{
  /**
   * @brief When calling a get function, say as in 
   * loss += myTensor[i], then we need to build a graph in between 
   * the myTensor[i] and the myTensor object. Hence this node.
   * 
   */
  class GetterNode final : public GraphNode {
    using multiDimIdx_t = std::vector<tensorDim_t>;

    private:
      const std::variant<tensorSize_t, multiDimIdx_t> idx;

    public:
      explicit GetterNode(std::shared_ptr<Tensor> t, const tensorSize_t idx) 
        : GraphNode({std::move(t)}), idx{idx} {}

      explicit GetterNode(std::shared_ptr<Tensor> t, const multiDimIdx_t& idx) 
        : GraphNode({std::move(t)}), idx{idx} {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };}
