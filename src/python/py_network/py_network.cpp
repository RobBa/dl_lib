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
  // TODO

  // Layers
  class_<layers::LayerBase, boost::noncopyable>("LayerBase", no_init)
    // attributes
    .add_property("dims", make_function(&layers::LayerBase::getDims, return_internal_reference<>()))
    .add_property("weights", make_function(&layers::LayerBase::getWeights))
    .add_property("bias", make_function(&layers::LayerBase::getBias))
    // methods
    .def("addActivation", make_function(&layers::LayerBase::addActivation))    
  ;

  class_<layers::FfLayer, std::shared_ptr<layers::FfLayer>, bases<layers::LayerBase>, boost::noncopyable>("FfLayer", no_init)
    // init
    .def(init<const std::vector<tensorDim_t>&>())
    .def(init<const std::vector<tensorDim_t>&, bool>())
    .def(init<const std::vector<tensorDim_t>&, bool, bool>())
    .def(init<const std::vector<tensorDim_t>&, Device>())
    .def(init<const std::vector<tensorDim_t>&, Device, bool>())
    .def(init<const std::vector<tensorDim_t>&, Device, bool, bool>())
    // methods
    .def("forward", WRAP_METHOD_ONE_TENSORARG(layers::FfLayer, Py_Network::ffForward))
    // operators
    .def("__str__", &toString<layers::FfLayer>)
  ;

  class_<activation::ReLu, std::shared_ptr<activation::ReLu>, boost::noncopyable>("ReLU")
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(activation::ReLu, Py_Network::reluF))
    .def("__str__", &toString<activation::ReLu>)
  ;

  class_<activation::LeakyReLu, std::shared_ptr<activation::LeakyReLu>, boost::noncopyable>("LeakyReLU", init<ftype>())
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(activation::LeakyReLu, Py_Network::leakyReluF))
    .def("__str__", &toString<activation::LeakyReLu>)
  ;

  class_<activation::Softmax, std::shared_ptr<activation::Softmax>, boost::noncopyable>("Softmax")
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(activation::Softmax, Py_Network::softmaxF))
    .def("__str__", &toString<activation::Softmax>)
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

  // Trainers
  // TODO
}

/*
ftype Py_Layers::layerGetItem(const layers::LayerBase& self, boost::python::object index) {
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

void Py_Layers::layerSetItem(layers::LayerBase& self, boost::python::object index, ftype value) {
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