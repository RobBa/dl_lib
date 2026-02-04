/**
 * @file topological_sort.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-03
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "topological_sort.h"
#include "tensor.h"

#include <queue>
#include <utility>

using namespace std;
using namespace graph;

/**
 * @brief Sorts the computational graph from end to beginning.
 * 
 * @param root The last node from which backward is started.
 */
vector< Tensor* > TopologicalSort::reverseSort(Tensor* root) {
  unordered_map<Tensor*, size_t> edgeCounts;

  // pass 1: BFS to get number of parent nodes per node
  queue< Tensor* > nodeQueue;
  nodeQueue.push(root);

  while(!nodeQueue.empty()){
    const auto& tensorPtr = nodeQueue.front();
    const auto& parents = tensorPtr->cgNode->parents;

    edgeCounts[tensorPtr] = parents.size();
    for(const auto& parent: parents){
      nodeQueue.push(parent);
    }

    nodeQueue.pop();
  }

  // pass 2: topological sort based on Kahn's algorithm
  vector< Tensor* > res; // TODO: reserve capacity to save runtime?
  nodeQueue.push(root);
  while(!nodeQueue.empty()){
    auto tensorPtr = nodeQueue.front();
    nodeQueue.pop();

    const auto& parents = tensorPtr->cgNode->parents;
    for(const auto& parent: parents){
      edgeCounts[parent]--;
      if(edgeCounts[parent]==0){
        nodeQueue.push(parent);
      }
    }

    res.push_back(move(tensorPtr));
  }

  return res;
}