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
#include <unordered_set>
#include <stack>

#include <utility>

using namespace std;
using namespace graph;

#ifndef NDEBUG
/**
 * @brief reverseSort() implementation expects a DAG. Because graph is directed 
 * by definition (only links to parents), we only need to check for cycles, otherwise
 * toposort will run in an infinite loop.
 */
bool TopologicalSort::hasCycles(const Tensor* root) {
  assert(root->cgNode);

  // DFS method to detect cycle from start to itself
  auto checkNodeHasCycle = [](const Tensor* start) -> bool {
    assert(start->cgNode);

    stack<const Tensor*> tStack;

    auto pushParentsOnStack = [&tStack](const Tensor* t){
      for(auto parent: t->cgNode->getParents()){
        if(parent->cgNode){
          tStack.push(parent.get());
        }
      }
    };

    pushParentsOnStack(start);

    while(!tStack.empty()){
      auto t = tStack.top();
      assert(t->cgNode);

      if(t==start){
        return true;
      }
      tStack.pop();
      
      pushParentsOnStack(t);
    }

    return false;
  };

  unordered_set<const Tensor*> visited;
  queue<const Tensor*> toCheckTensors;
  toCheckTensors.push(root);

  while(!toCheckTensors.empty()){
    auto t = toCheckTensors.front();
    assert(t->cgNode);

    if(checkNodeHasCycle(t)){
      return true;
    }
    toCheckTensors.pop();

    visited.insert(t);
    for(auto parent: t->cgNode->getParents()) {
      const auto parentPtr = parent.get();

      if(parent->cgNode && !visited.contains(parentPtr)){
        toCheckTensors.push(parent.get());
      }
    }
  }
  
  return false;
}
#endif // NDEBUG

/**
 * @brief Sorts the computational graph from end to beginning.
 * 
 * @param root The last node in graph, from which backward is started.
 */
vector< Tensor* > TopologicalSort::reverseSort(Tensor* root) {
  assert(!hasCycles(root));

  if(!root->cgNode){
    __throw_invalid_argument("Trying topo-sort on node that is not part of graph");
  }

  unordered_map<Tensor*, size_t> edgeCounts;

  // pass 1: BFS to get number of parent nodes per node
  queue< Tensor* > nodeQueue;
  nodeQueue.push(root);
  edgeCounts[root] = 0;

  auto updateQueueAndEdgeCounts = [&nodeQueue, &edgeCounts](Tensor* t){
    if(!edgeCounts.contains(t)) {
      edgeCounts[t] = 1;
      nodeQueue.push(t); // each node appears only once
    }
    else{
      edgeCounts[t]++;
    }
  };

  while(!nodeQueue.empty()){
    const auto tensorPtr = nodeQueue.front();
    nodeQueue.pop();

    for(const auto& parent: tensorPtr->cgNode->getParents()){
      if(parent->cgNode){
        updateQueueAndEdgeCounts(parent.get());
      }
    }
  }

  auto pushParentsWithGraphNode = [&nodeQueue, &edgeCounts](Tensor* t){
    const auto& parents = t->cgNode->getParents();
    for(const auto& parent: parents){ // TODO: check for requiresGrad to save runtime?
      if(!parent->cgNode)
        continue;

      auto parentPtr = parent.get();
      edgeCounts[parentPtr]--;
      if(edgeCounts[parentPtr]==0){
        nodeQueue.push(parentPtr);
      }
    }
  };

  // pass 2: topological sort based on Kahn's algorithm
  vector< Tensor* > res; // TODO: reserve capacity to save runtime?
  nodeQueue.push(root);
  while(!nodeQueue.empty()){
    auto tensorPtr = nodeQueue.front();
    nodeQueue.pop();

    if(edgeCounts[tensorPtr]==0){
      pushParentsWithGraphNode(tensorPtr);
      res.push_back(tensorPtr);
    }
    else {
      nodeQueue.push(tensorPtr);
    }    
  }

  return res;
}