/**
 * @file base_train_loop.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"
#include "module/module_base.h"

#include "training/optimizers/optimizer_base.h"
#include "training/loss_functions/loss_base.h"

#include <memory>
#include <utility>

namespace train {
  class BaseTrainLoop {
    protected:
      ftype lr;

      const size_t epochs;
      const tensorDim_t bsize;

      std::shared_ptr<LossBase> loss;
      std::shared_ptr<OptimizerBase> optim;
      std::shared_ptr<module::ModuleBase> graph;

    public:
      BaseTrainLoop(std::shared_ptr<module::ModuleBase>& graph, std::shared_ptr<LossBase> loss,
                  std::shared_ptr<OptimizerBase> optim, ftype lr, size_t epochs, tensorDim_t bsize) 
        : graph{std::move(graph)}, optim{std::move(optim)}, loss{loss}, lr{lr}, epochs{epochs}, bsize{bsize} {
          
        };
      
      ~BaseTrainLoop() noexcept = default;

      BaseTrainLoop(const BaseTrainLoop& other) = delete;
      BaseTrainLoop& operator=(const BaseTrainLoop& other) = delete;

      BaseTrainLoop(BaseTrainLoop&& other) noexcept = default;
      BaseTrainLoop& operator=(BaseTrainLoop&& other) noexcept = default;

      void run(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& y, const bool shuffle);
  };
}