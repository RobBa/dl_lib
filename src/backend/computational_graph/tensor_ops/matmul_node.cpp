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
using namespace cgraph;

vector<shared_ptr<Tensor>> MatMulNode::backward(const Tensor& upstreamGrad) {
    assert(!upstreamGrad.getRequiresGrad());
    // TODO: optimize operators
    return {
        make_shared<Tensor>(upstreamGrad.matmul(*parents[1], false, true)), 
        make_shared<Tensor>(parents[0]->matmul(upstreamGrad, true, false))
    };
}