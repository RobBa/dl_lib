/**
 * @file train_mode.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "training/optimizers/optimizer_base.h"
#include "training/loss_functions/loss_base.h"

#include "data_modeling/tensor.h"
#include "training/loss_functions/loss_base.h"
#include "networks/sequential.h"

#include <memory>
#include <utility>

namespace train {
  class BaseTrainer {
    protected:
      ftype lr;

      const size_t epochs;
      const tensorDim_t bsize;

      std::shared_ptr<LossBase> loss;
      std::shared_ptr<OptimizerBase> optim;
      std::shared_ptr<SequentialNetwork> network;

    public:
      BaseTrainer(std::shared_ptr<SequentialNetwork>& network, std::shared_ptr<LossBase> loss,
                  std::shared_ptr<OptimizerBase> optim, ftype lr, size_t epochs, tensorDim_t bsize) 
        : network{std::move(network)}, optim{std::move(optim)}, loss{loss}, lr{lr}, epochs{epochs}, bsize{bsize} {};
      
      ~BaseTrainer() noexcept = default;

      BaseTrainer(const BaseTrainer& other) = delete;
      BaseTrainer& operator=(const BaseTrainer& other) = delete;

      BaseTrainer(BaseTrainer&& other) noexcept = default;
      BaseTrainer& operator=(BaseTrainer&& other) noexcept = default;

      void run(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& y, const bool shuffle);
  };
}