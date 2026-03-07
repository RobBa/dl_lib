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

#include "computational_graph/graph_node.h"

namespace graph {
  class AddNode final : public GraphNode {
    private:
      // if t2 has been a vector we broadcast t2 into t1, see Tensor::add()
      bool broadcasted = false;

    public:
      explicit AddNode(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) 
        : GraphNode({std::move(t1), std::move(t2)}) {
          // t2 is either tensor of same size or 1D-vector as bias
          assert(t1->getDims().nDims()>=t2->getDims().nDims());

          broadcasted = parents[0]->getDims() != parents[1]->getDims();
        }

      std::vector<std::shared_ptr<Tensor>> backward(const Tensor& upstreamGrad) override;
  };
}