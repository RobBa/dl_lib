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
    res->setCgNode(make_shared<graph::ElementwiseMulNode>(left, right));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> graph::add(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>(*left + *right);
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    res->setCgNode(make_shared<graph::AddNode>(left, right));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> graph::matmul(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>(left->matmul(*right));
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    res->setCgNode(make_shared<graph::MatMulNode>(left, right));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> graph::mul(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) * scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<graph::ScalarMulNode>(t, scalar));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> graph::mul(ftype scalar, const shared_ptr<Tensor> t) {
  return graph::mul(t, scalar);
}

shared_ptr<Tensor> graph::add(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) + scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<graph::ScalarAddNode>(t));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> graph::add(ftype scalar, const shared_ptr<Tensor> t) {
  return graph::add(t, scalar);
}

shared_ptr<Tensor> graph::sub(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) - scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<graph::ScalarAddNode>(t));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> graph::div(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) / scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<graph::ScalarMulNode>(t, 1 / scalar));
    assert(res->getRequiresGrad());
  }
  return res;
}

/**
 * @brief Special linear indexing, see getItem() overloads in tensor. 
 * Used to keep the computational graph intact.
 * E.g. if we have something like 
 * 
 * loss = loss + other.get(i), we need to make sure get(i) can map to computational graph.
 */
shared_ptr<Tensor> graph::getAsShared(const shared_ptr<Tensor>& t, tensorSize_t idx) {
  ftype val = t->getItem(idx);
  return make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val}, 
                             t->getDevice(), t->getRequiresGrad()); 
}

/**
 * @brief Special linear indexing, see getItem() overloads in tensor. 
 * Used to keep the computational graph intact.
 * E.g. if we have something like 
 * 
 * loss = loss + other.get(i), we need to make sure get(i) can map to computational graph.
 */
std::shared_ptr<Tensor> graph::getAsShared(const Tensor& t, tensorSize_t idx) {
  ftype val = t.getItem(idx);
  return make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val}, 
                             t.getDevice(), t.getRequiresGrad()); 
}

/**
 * @brief Used to keep the computational graph intact.
 * E.g. if we have something like 
 * 
 * loss = loss + other.get(i), we need to make sure get(i) can map to computational graph.
 */
shared_ptr<Tensor> graph::getAsShared(const shared_ptr<Tensor>& t, vector<tensorDim_t>&& idx) {
  ftype val = t->getItem(std::move(idx));
  return make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val}, 
                             t->getDevice(), t->getRequiresGrad()); 
}

/**
 * @brief Used to keep the computational graph intact.
 * E.g. if we have something like 
 * 
 * loss = loss + other.get(i), we need to make sure get(i) can map to computational graph.
 */
std::shared_ptr<Tensor> graph::getAsShared(const Tensor& t, std::vector<tensorDim_t>&& idx) {
  ftype val = t.getItem(std::move(idx));
  return make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val}, 
                             t.getDevice(), t.getRequiresGrad()); 
}