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

#include <memory>

namespace graph {
  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right);

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right);

  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right);

  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> left, ftype scalar); 
  std::shared_ptr<Tensor> mul(ftype scalar, const std::shared_ptr<Tensor> left); 

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> left, ftype scalar);    
  std::shared_ptr<Tensor> add(ftype scalar, const std::shared_ptr<Tensor> left);

  std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> left, ftype scalar);
  std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> left, ftype scalar);

  std::shared_ptr<Tensor> get(const std::shared_ptr<Tensor>& t, tensorSize_t idx);
  std::shared_ptr<Tensor> get(const std::shared_ptr<Tensor>& t, const std::vector<tensorDim_t>& idx);
}
 