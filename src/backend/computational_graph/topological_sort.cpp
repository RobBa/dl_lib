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
#include "data_modeling/tensor.h"

#include <queue>
#include <unordered_map>

#include <utility>

using namespace std;
using namespace graph;

/**
 * @brief Sorts the computational graph from end to beginning.
 * 
 * @param root The last node in graph, from which backward is started.
 */
vector< Tensor* > TopologicalSort::reverseSort(Tensor* root) {
  if(!root->cgNode){
    __throw_invalid_argument("Trying topo-sort on node that is not part of graph");
  }

  unordered_map<Tensor*, size_t> edgeCounts;

  // pass 1: BFS to get number of parent nodes per node
  queue< Tensor* > nodeQueue;
  nodeQueue.push(root);

  while(!nodeQueue.empty()){
    const auto tensorPtr = nodeQueue.front();
    nodeQueue.pop();

    const auto& parents = tensorPtr->cgNode->getParents();
    edgeCounts[tensorPtr] = parents.size();
    for(const auto& parent: parents){
      if(parent->cgNode){
        nodeQueue.push(parent.get());
      }
    }
  }

  // pass 2: topological sort based on Kahn's algorithm
  vector< Tensor* > res; // TODO: reserve capacity to save runtime?
  nodeQueue.push(root);
  while(!nodeQueue.empty()){
    auto tensorPtr = nodeQueue.front();
    nodeQueue.pop();

    const auto& parents = tensorPtr->cgNode->getParents();
    for(const auto& parent: parents){ // TODO: check for requiresGrad to save runtime?
      if(!parent->cgNode)
        continue;
      
      edgeCounts[parent.get()]--;
      if(edgeCounts[parent.get()]==0){
        nodeQueue.push(parent.get());
      }
    }

    res.push_back(tensorPtr);
  }

  return res;
}