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
#include "getter_node.h"

using namespace std;

shared_ptr<Tensor> cgraph::mul(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>((*left) * (*right));
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    res->setCgNode(make_shared<cgraph::ElementwiseMulNode>(left, right));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> cgraph::add(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>(*left + *right);
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    res->setCgNode(make_shared<cgraph::AddNode>(left, right));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> cgraph::matmul(const shared_ptr<Tensor> left, const shared_ptr<Tensor> right) {
  auto res = make_shared<Tensor>(left->matmul(*right));
  if(left->getRequiresGrad() || right->getRequiresGrad()){
    res->setCgNode(make_shared<cgraph::MatMulNode>(left, right));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> cgraph::mul(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) * scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<cgraph::ScalarMulNode>(t, scalar));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> cgraph::mul(ftype scalar, const shared_ptr<Tensor> t) {
  return cgraph::mul(t, scalar);
}

shared_ptr<Tensor> cgraph::add(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) + scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<cgraph::ScalarAddNode>(t));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> cgraph::add(ftype scalar, const shared_ptr<Tensor> t) {
  return cgraph::add(t, scalar);
}

shared_ptr<Tensor> cgraph::sub(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) - scalar);
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<cgraph::ScalarAddNode>(t));
    assert(res->getRequiresGrad());
  }
  return res;
}

shared_ptr<Tensor> cgraph::div(const shared_ptr<Tensor> t, ftype scalar) {
  auto res = make_shared<Tensor>((*t) / scalar);
  if(t->getRequiresGrad()){
    constexpr ftype eps = 1e-6;
    res->setCgNode(std::make_shared<cgraph::ScalarMulNode>(t, 1/std::max(scalar, eps)));
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
shared_ptr<Tensor> cgraph::get(const shared_ptr<Tensor>& t, tensorSize_t idx) {
  ftype val = t->getItem(idx);
  auto res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val}, 
                             t->getDevice());
                             
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<cgraph::GetterNode>(t, idx));
    assert(res->getRequiresGrad());
  }
  return res;
}

/**
 * @brief Used to keep the computational graph intact.
 * E.g. if we have something like 
 * 
 * loss = loss + other.get(i), we need to make sure get(i) can map to computational graph.
 */
shared_ptr<Tensor> cgraph::get(const shared_ptr<Tensor>& t, const vector<tensorDim_t>& idx) {
  ftype val = t->getItem(std::move(idx));
  auto res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val}, 
                             t->getDevice());
  if(t->getRequiresGrad()){
    res->setCgNode(std::make_shared<cgraph::GetterNode>(t, idx));
    assert(res->getRequiresGrad());
  }
  return res;
}

/**
 * @brief Takes the sum of the whole tensor, then returns result as vector.
 */
shared_ptr<Tensor> cgraph::sumTensor(const shared_ptr<Tensor> t) {
  auto res = make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{0.0}, 
                                 t->getDevice(), t->getRequiresGrad());
  for(tensorSize_t i=0; i<t->getSize(); i++){
    res = cgraph::add(res, cgraph::get(t, i));
  }
  return res;
}