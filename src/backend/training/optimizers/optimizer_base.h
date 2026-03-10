/**
 * @file optimizer_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"
#include "training/loss_functions/loss_base.h"

#include <memory>
#include <utility>

namespace train {
  class OptimizerBase {
    protected:
      ftype lr;

      const size_t epochs;
      const tensorDim_t bsize;

      std::shared_ptr<LossBase> loss;
      std::vector< std::shared_ptr<Tensor> > params;

      virtual void step(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) = 0;

    public:
      OptimizerBase(std::vector< std::shared_ptr<Tensor> >& params, std::shared_ptr<LossBase> loss,
                    ftype lr, size_t epochs, tensorDim_t bsize) 
        : params{std::move(params)}, loss{loss}, lr{lr}, epochs{epochs}, bsize{bsize} {};
      
      ~OptimizerBase() noexcept = default;

      OptimizerBase(const OptimizerBase& other) = delete;
      OptimizerBase& operator=(const OptimizerBase& other) = delete;

      OptimizerBase(OptimizerBase&& other) noexcept = default;
      OptimizerBase& operator=(OptimizerBase&& other) noexcept = default;

      void run(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& y, const bool shuffle);
  };
}