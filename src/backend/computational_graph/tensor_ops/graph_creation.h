/**
 * @file graph_creation.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Tensor operations that actually create the computational graph.
 * @version 0.1
 * @date 2026-02-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "data_modeling/tensor.h"

#include "add_node.h"
#include "matmul_node.h"
#include "elementwise_mul_node.h"
#include "scalar_op_nodes.h"
#include "getter_node.h"

#include <memory>
#include <algorithm>

namespace cgraph {

  // Arithmetic operations

  inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right) {
    auto res = std::make_shared<Tensor>((*left) * (*right));
    if(left->getRequiresGrad() || right->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::ElementwiseMulNode>(left, right));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right) {
    auto res = std::make_shared<Tensor>(*left + *right);
    if(left->getRequiresGrad() || right->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::AddNode>(left, right));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right) {
    auto res = std::make_shared<Tensor>(left->matmul(*right));
    if(left->getRequiresGrad() || right->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::MatMulNode>(left, right));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> t, ftype scalar) {
    auto res = std::make_shared<Tensor>((*t) * scalar);
    if(t->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::ScalarMulNode>(t, scalar));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> mul(ftype scalar, const std::shared_ptr<Tensor> t) {
    return cgraph::mul(t, scalar);
  }

  inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> t, ftype scalar) {
    auto res = std::make_shared<Tensor>((*t) + scalar);
    if(t->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::ScalarAddNode>(t));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> add(ftype scalar, const std::shared_ptr<Tensor> t) {
    return cgraph::add(t, scalar);
  }

  inline std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> t, ftype scalar) {
    auto res = std::make_shared<Tensor>((*t) - scalar);
    if(t->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::ScalarAddNode>(t));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> t, ftype scalar) {
    auto res = std::make_shared<Tensor>((*t) / scalar);
    if(t->getRequiresGrad()){
      constexpr ftype eps = 1e-9;
      res->setCgNode(std::make_shared<cgraph::ScalarMulNode>(t, 1/std::max(scalar, eps)));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  // Getter methods — keep the computational graph intact for indexing operations

  inline std::shared_ptr<Tensor> get(const std::shared_ptr<Tensor>& t, tensorSize_t idx) {
    ftype val = t->get(idx);
    auto res = std::make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val},
                                        t->getDevice());
    if(t->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::GetterNode>(t, idx));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  inline std::shared_ptr<Tensor> get(const std::shared_ptr<Tensor>& t, const std::vector<tensorDim_t>& idx) {
    ftype val = t->get(std::move(idx));
    auto res = std::make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{val},
                                        t->getDevice());
    if(t->getRequiresGrad()){
      res->setCgNode(std::make_shared<cgraph::GetterNode>(t, idx));
      assert(res->getRequiresGrad());
    }
    return res;
  }

  // Composite operations

  inline std::shared_ptr<Tensor> sumTensor(const std::shared_ptr<Tensor> t) {
    auto res = std::make_shared<Tensor>(std::vector<tensorDim_t>{1}, std::vector<ftype>{0.0},
                                        t->getDevice(), t->getRequiresGrad());
    for(tensorSize_t i=0; i<t->getSize(); i++){
      res = cgraph::add(res, cgraph::get(t, i));
    }
    return res;
  }

}
