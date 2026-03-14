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

#include "computational_graph/graph_node.h"
#include "data_modeling/tensor.h"

#include <memory>
#include <utility>

namespace cgraph {
  class SigmoidNode final : public GraphNode {
    private:
      // cache the result of the forward function
      std::shared_ptr<const Tensor> sigmoid;

    public:
      explicit SigmoidNode(std::shared_ptr<Tensor> t, std::shared_ptr<const Tensor> sigmoid) 
        : GraphNode({std::move(t)}), sigmoid{std::move(sigmoid)} {}

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}
