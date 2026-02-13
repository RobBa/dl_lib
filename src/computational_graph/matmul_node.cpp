/**
 * @file matmul_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "matmul_node.h"

using namespace std;
using namespace graph;

vector<shared_ptr<Tensor>> MatMulNode::backward(const Tensor& upstreamGrad) {
    __throw_runtime_error("Not implemented yet");
}