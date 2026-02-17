/**
 * @file graph_creation.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-17
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "graph_creation.h"

#include "add_node.h"
#include "matmul_node.h"
#include "elementwise_mul_node.h"
#include "scalar_op_nodes.h"

using namespace std;

shared_ptr<Tensor> graph::mul(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>((*left) * (*right));
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(make_shared<graph::ElementwiseMulNode>(left, right));
  }
  return res;
}

shared_ptr<Tensor> graph::add(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>(*left + *right);
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(make_shared<graph::AddNode>(left, right));
  }
  return res;
}

shared_ptr<Tensor> graph::matmul(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>(left->matmul(*right));
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(make_shared<graph::MatMulNode>(left, right));
  }
  return res;
}

shared_ptr<Tensor> graph::mul(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) * scalar);
  if(t->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(std::make_shared<graph::ScalarMulNode>(t, scalar));
  }
  return res;
}

shared_ptr<Tensor> graph::mul(ftype scalar, const shared_ptr<Tensor> t) {
  return graph::mul(t, scalar);
}

shared_ptr<Tensor> graph::add(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) + scalar);
  if(t->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(std::make_shared<graph::ScalarAddNode>(t));
  }
  return res;
}

shared_ptr<Tensor> graph::add(ftype scalar, const shared_ptr<Tensor> t) {
  return graph::add(t, scalar);
}

shared_ptr<Tensor> graph::sub(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) - scalar);
  if(t->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(std::make_shared<graph::ScalarAddNode>(t));
  }
  return res;
}

shared_ptr<Tensor> graph::div(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) / scalar);
  if(t->getRequiresGrad()){
    assert(res->getRequiresGrad());
    res->setCgNode(std::make_shared<graph::ScalarMulNode>(t, 1 / scalar));
  }
  return res;
}