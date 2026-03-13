/**
 * @file layers.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-11-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "py_network_util.h"
#include "python_templates.h"
#include "utility/global_params.h"

#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include "training/optimizers/sgd.h"
#include "training/optimizers/rmsprop.h"

#include "training/trainers/base_train_loop.h"

#include <stdexcept>

BOOST_PYTHON_MODULE(py_layers)
{
  using namespace Py_Util;
  
  using namespace boost::python;

  #define WRAP_METHOD_ONE_TENSORARG(T, method) \
  +[](const T& self, Tensor& t) -> std::shared_ptr<Tensor> { \
    return (self.*method)(t.getSharedPtr()); \
  }

  #define WRAP_METHOD_TWO_TENSORARGS(T, method) \
  +[](const T& self, Tensor& t1, Tensor& t2) -> std::shared_ptr<Tensor> { \
    return (self.*method)(t1.getSharedPtr(), t2.getSharedPtr()); \
  }

  // Networks
  class_<Py_Network::ModuleBaseWrapper, std::shared_ptr<networks::Sequential>, boost::noncopyable>("Module", no_init)
    // operators
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::ModuleBaseWrapper, Py_Network::moduleForward))
    .def("__str__", &toString<module::FfLayer>)
  ;

  class_<module::FfLayer, std::shared_ptr<module::FfLayer>, boost::noncopyable>("FfLayer", no_init)
    // init
    .def(init<const std::vector<tensorDim_t>&>())
    .def(init<const std::vector<tensorDim_t>&, bool>())
    .def(init<const std::vector<tensorDim_t>&, bool, bool>())
    .def(init<const std::vector<tensorDim_t>&, Device>())
    .def(init<const std::vector<tensorDim_t>&, Device, bool>())
    .def(init<const std::vector<tensorDim_t>&, Device, bool, bool>())
    // methods
    .add_property("dims", make_function(&module::FfLayer::getDims, return_internal_reference<>()))
    .add_property("weights", &module::FfLayer::getWeights)
    .add_property("bias", &module::FfLayer::getBias)
    .add_property("params", &module::ModuleBase::getParams)
        // operators
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::FfLayer, Py_Network::ffForward))
    .def("__str__", &toString<module::FfLayer>)
  ;

  class_<module::ReLu, std::shared_ptr<module::ReLu>, boost::noncopyable>("ReLU")
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::ReLu, Py_Network::reluF))
    .def("__str__", &toString<module::ReLu>)
  ;

  class_<module::LeakyReLu, std::shared_ptr<module::LeakyReLu>, boost::noncopyable>("LeakyReLU", init<ftype>())
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::LeakyReLu, Py_Network::leakyReluF))
    .def("__str__", &toString<module::LeakyReLu>)
  ;

  class_<module::Softmax, std::shared_ptr<module::Softmax>, boost::noncopyable>("Softmax")
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::Softmax, Py_Network::softmaxF))
    .def("__str__", &toString<module::Softmax>)
  ;

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

/*
ftype Py_module::layerGetItem(const module::ModuleBase& self, boost::python::object index) {
  extract<int> int_extractor(index);
        
  // Single integer index (1D)
  if(int_extractor.check()) {
      int i0 = int_extractor();
      return self.getItem(i0);
  }
        
  // Tuple index (2D, 3D, or 4D)
  extract<tuple> tuple_extractor(index);
  if(tuple_extractor.check()) {
      tuple idx_tuple = tuple_extractor();
      int ndim = boost::python::len(idx_tuple);
      
      if (ndim == 2) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        return self.getItem(i0, i1);
      }
      else if (ndim == 3) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        return self.getItem(i0, i1, i2);
      }
      else if (ndim == 4) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        int i3 = extract<int>(idx_tuple[3]);
        return self.getItem(i0, i1, i2, i3);
      }
      else {
        PyErr_SetString(PyExc_IndexError, "Unsupported number of dimensions");
        throw_error_already_set();
      }
  }
        
  PyErr_SetString(PyExc_TypeError, "Index must be an integer or tuple");
  throw_error_already_set();
  return 0.0; // Never reached
}

void Py_module::layerSetItem(module::ModuleBase& self, boost::python::object index, ftype value) {
  extract<int> int_extractor(index);
        
  // Single integer index (1D)
  if(int_extractor.check()) {
      int i0 = int_extractor();
      self.setItem(value, i0);\
      return;
  }
        
  // Tuple index (2D, 3D, or 4D)
  extract<tuple> tuple_extractor(index);
  if(tuple_extractor.check()) {
      tuple idx_tuple = tuple_extractor();
      int ndim = boost::python::len(idx_tuple);
      
      if (ndim == 2) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        self.setItem(value, i0, i1);
      }
      else if (ndim == 3) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        self.setItem(value, i0, i1, i2);
      }
      else if (ndim == 4) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        int i3 = extract<int>(idx_tuple[3]);
        self.setItem(value, i0, i1, i2, i3);
      }
      else {
        PyErr_SetString(PyExc_IndexError, "Unsupported number of dimensions");
        throw_error_already_set();
      }
      return;
  }
        
  PyErr_SetString(PyExc_TypeError, "Index must be an integer or tuple");
  throw_error_already_set();
}*/