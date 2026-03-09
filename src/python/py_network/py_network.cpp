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

#include "layers/ff_layer.h"

#include "activation_functions/relu.h"
#include "activation_functions/leaky_relu.h"
#include "activation_functions/softmax.h"

#include "training/loss_functions/bce_loss.h"
#include "training/loss_functions/crossentropy_loss.h"

#include "training/optimizers/sgd.h"

#include <stdexcept>

BOOST_PYTHON_MODULE(py_layers)
{
  using namespace std;

  using namespace Py_Util;
  using namespace Py_Network;
  
  using namespace boost::python;

  // Layers
  class_<LayerBaseWrap, boost::noncopyable>("LayerBase", no_init)
    // attributes
    .add_property("dims", make_function(&layers::LayerBase::getDims, return_internal_reference<>()))
    .add_property("weights", make_function(&layers::LayerBase::getWeights))
    .add_property("bias", make_function(&layers::LayerBase::getBias))
    // methods
    .def("forward", pure_virtual(Py_Network::layerforward))
    .def("addActivation", make_function(&layers::LayerBase::addActivation))
    // operators
    .def("__str__", &toString<layers::LayerBase>)
  ;

  class_<layers::FfLayer, bases<LayerBaseWrap>, boost::noncopyable>("FfLayer", no_init)
    .def(init<const std::vector<tensorDim_t>&, optional<bool>, optional<bool> >())
    .def(init<const std::vector<tensorDim_t>&, Device, optional<bool>, optional<bool> >())
    .def("forward", &layers::FfLayer::forward)
  ;

  // Activation functions
  class_<ActivationFunctionWrap, boost::noncopyable>("ActivationFunctionBase", no_init)
    .def("call", pure_virtual(&ActivationFunctionWrap::operator()))
    .def("__str__", &toString<activation::ActivationFunctionBase>)
  ;

  class_<activation::ReLu, std::shared_ptr<ActivationFunctionWrap>, bases<ActivationFunctionWrap> >("ReLU", init)
    .def("call", &activation::ReLu::operator())
  ;

  class_<activation::LeakyReLu, std::shared_ptr<ActivationFunctionWrap>, bases<ActivationFunctionWrap> >("LeakyReLU", init<ftype>)
    .def("call", &activation::LeakyReLu::operator())
  ;

  class_<activation::Softmax, std::shared_ptr<ActivationFunctionWrap>, bases<ActivationFunctionWrap> >("Softmax", init)
    .def("call", &activation::Softmax::operator())
  ;

  // Loss functions
  class_<LossWrap, boost::noncopyable>("LossBase", no_init)
    .def("call", pure_virtual(&LossWrap::operator()))
  ;

  class_<train::BceLoss, boost::noncopyable>("BCE", no_init)
    .def("call", pure_virtual(&train::BceLoss::operator()))
  ;

  class_<train::CrossEntropyLoss, boost::noncopyable>("CrossEntropy", no_init)
    .def("call", pure_virtual(&train::CrossEntropyLoss::operator()))
  ;

  // Optimizers
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