/**
 * @file scalar_op_nodes.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-17
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "computational_graph/graph_node.h"

namespace graph {
  class ScalarAddNode final : public GraphNode {  
    public:
      explicit ScalarAddNode(std::shared_ptr<Tensor> t) 
        : GraphNode({std::move(t)}) {}

      ScalarAddNode(const ScalarAddNode& other) = delete;
      ScalarAddNode& operator=(const ScalarAddNode& other) = delete;

      ScalarAddNode(ScalarAddNode&& other) = default;
      ScalarAddNode& operator=(ScalarAddNode&& other) = default;

      ~ScalarAddNode() noexcept = default; 

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };

  class ScalarMulNode final : public GraphNode {
    private:
      const ftype factor;
    
    public:
      explicit ScalarMulNode(std::shared_ptr<Tensor> t, ftype factor) 
        : GraphNode({std::move(t)}), factor{factor} {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}