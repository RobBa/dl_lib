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
#include "training/loss_functions/bce_sigmoid_loss.h"
#include "training/loss_functions/crossentropy_loss.h"
#include "training/loss_functions/crossentropy_softmax_loss.h"

#include "training/optimizers/optimizer_base.h"
#include "training/optimizers/sgd.h"
#include "training/optimizers/rmsprop.h"

#include "training/trainers/base_train_loop.h"

BOOST_PYTHON_MODULE(_train)
{
  // enable conversion from Tensor registered in _core
  boost::python::object coreModule = boost::python::import("dl_lib._compiled._core");
  boost::python::register_ptr_to_python<std::shared_ptr<Tensor>>();

  using namespace boost::python;

  // Loss functions
  class_<train::BceLoss, std::shared_ptr<train::BceLoss>, boost::noncopyable>("BCE")
    .def("__call__", &train::BceLoss::operator())
  ;

  class_<train::BceSigmoidLoss, std::shared_ptr<train::BceSigmoidLoss>, boost::noncopyable>("BceWithSigmoid")
      .def("__call__", &train::BceSigmoidLoss::operator())
  ;

  class_<train::CrossEntropyLoss, std::shared_ptr<train::CrossEntropyLoss>, boost::noncopyable>("CrossEntropy")
    .def("__call__", &train::CrossEntropyLoss::operator())
  ;

  class_<train::CrossEntropySoftmaxLoss, std::shared_ptr<train::CrossEntropySoftmaxLoss>, boost::noncopyable>("CrossEntropyWithSoftmax")
      .def("__call__", &train::CrossEntropySoftmaxLoss::operator())
  ;

  // Optimizers
  class_<train::OptimizerBase, boost::noncopyable>("_OptimizerBase", no_init)
      .def("step", pure_virtual(&train::OptimizerBase::step))
      .def("zeroGrad", &train::OptimizerBase::zeroGrad)
      .def("clipGradients", &train::OptimizerBase::clipGradients)
  ;

  class_<train::SgdOptimizer, bases<train::OptimizerBase>, std::shared_ptr<train::SgdOptimizer>, boost::noncopyable>("SGD", no_init)
      .def(init<std::vector<std::shared_ptr<Tensor>>, ftype>())
      .def("step", &train::SgdOptimizer::step)
  ;

  class_<train::RmsPropOptimizer, bases<train::OptimizerBase>, std::shared_ptr<train::RmsPropOptimizer>, boost::noncopyable>("RmsProp", no_init)
      .def(init<std::vector<std::shared_ptr<Tensor>>, ftype, ftype>())
      .def("step", &train::RmsPropOptimizer::step)
  ;

  // Trainers
  class_<train::BaseTrainLoop, std::shared_ptr<train::BaseTrainLoop>, boost::noncopyable>("TrainLoop", no_init)
      .def(init<std::shared_ptr<module::ModuleBase>&, std::shared_ptr<train::LossBase>,
                std::shared_ptr<train::OptimizerBase>, size_t, tensorDim_t>())
      .def("run", &train::BaseTrainLoop::run)
  ;
}