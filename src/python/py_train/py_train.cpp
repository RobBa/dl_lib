/**
 * @file py_train.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <boost/python.hpp>

#include "utility/global_params.h"

#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include "training/optimizers/sgd.h"
#include "training/optimizers/rmsprop.h"

#include "training/trainers/base_train_loop.h"

BOOST_PYTHON_MODULE(_train)
{
  using namespace boost::python;

  // Loss functions
  class_<train::BceLoss, std::shared_ptr<train::BceLoss>, boost::noncopyable>("BCE")
    .def("__call__", &train::BceLoss::operator())
  ;

  class_<train::CrossEntropyLoss, std::shared_ptr<train::CrossEntropyLoss>, boost::noncopyable>("CrossEntropy")
    .def("__call__", &train::CrossEntropyLoss::operator())
  ;

  // Optimizers
  class_<train::SgdOptimizer, std::shared_ptr<train::SgdOptimizer>, boost::noncopyable>("SGD", no_init)
    .def(init<std::vector< std::shared_ptr<Tensor> >, ftype>())
    .def("step", &train::SgdOptimizer::step)
  ;

  class_<train::RmsPropOptimizer, std::shared_ptr<train::RmsPropOptimizer>, boost::noncopyable>("RmsProp", no_init)
    .def(init<std::vector< std::shared_ptr<Tensor> >, ftype, ftype>())
    .def("step", &train::RmsPropOptimizer::step)
  ;

  // Trainers
  class_<train::BaseTrainLoop, std::shared_ptr<train::BaseTrainLoop>, boost::noncopyable>("TrainLoop", no_init)
    .def(init<std::shared_ptr<module::ModuleBase>&, std::shared_ptr<train::LossBase>, std::shared_ptr<train::OptimizerBase>,
              ftype, size_t, tensorDim_t>())
    .def("step", &train::RmsPropOptimizer::step)
  ;
}