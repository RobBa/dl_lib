/**
 * @file elementwise_mul_node.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "elementwise_mul_node.h"

using namespace std;
using namespace cgraph;

vector<shared_ptr<Tensor>> ElementwiseMulNode::backward(const Tensor& upstreamGrad) {
    assert(!upstreamGrad.getRequiresGrad());
    return {
        make_shared<Tensor>(upstreamGrad * (*parents[1])), 
        make_shared<Tensor>(upstreamGrad * (*parents[0]))
    };
}