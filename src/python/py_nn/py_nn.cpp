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

#include "py_nn_util.h"
#include "python_templates.h"
#include "utility/global_params.h"

#include <stdexcept>

BOOST_PYTHON_MODULE(_nn)
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
  class_<Py_nn::ModuleBaseWrapper, std::shared_ptr<Py_nn::ModuleBaseWrapper>, boost::noncopyable>("_Module", no_init)
    // methods  
    .def("_own_parameters", &module::ModuleBase::parameters)
    // operators
    .def("forward", pure_virtual(WRAP_METHOD_ONE_TENSORARG(Py_nn::ModuleBaseWrapper, Py_nn::moduleForward)))
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
    .add_property("params", &module::ModuleBase::parameters)
        // operators
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::FfLayer, Py_nn::ffForward))
    .def("__str__", &toString<module::FfLayer>)
  ;

  class_<module::ReLu, std::shared_ptr<module::ReLu>, boost::noncopyable>("ReLU")
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::ReLu, Py_nn::reluF))
    .def("__str__", &toString<module::ReLu>)
  ;

  class_<module::LeakyReLu, std::shared_ptr<module::LeakyReLu>, boost::noncopyable>("LeakyReLU", init<ftype>())
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::LeakyReLu, Py_nn::leakyReluF))
    .def("__str__", &toString<module::LeakyReLu>)
  ;

  class_<module::Softmax, std::shared_ptr<module::Softmax>, boost::noncopyable>("Softmax")
    .def("__call__", WRAP_METHOD_ONE_TENSORARG(module::Softmax, Py_nn::softmaxF))
    .def("__str__", &toString<module::Softmax>)
  ;
}