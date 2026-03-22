/**
 * @file py_data_modeling.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-21
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "data_modeling/tensor.h"

#include "py_core_util.h"
#include "python_templates.h"
#include "custom_converters.h"

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>

BOOST_PYTHON_MODULE(_core)
{
  using namespace boost::python;

  // some macros to make code below easier to read
  #define WRAP_TENSOR_METHOD_1(method) \
  +[](const Tensor& self, const Tensor& other) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>(self.method(other)); \
  }

  #define WRAP_SCALAR(method, T) \
  +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>(self.method(val)); \
  }

  #define WRAP_SCALAR_REVERSE(op, T) \
  +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>(val op self); \
  }

  // different, since those are not methods anymore
  #define WRAP_FREE_MEMBER_FUNC_1(fPtr, T1, T2) \
  +[](const Tensor& self, int v1, int v2) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((self.*fPtr)(v1, v2)); \
  }

  #define WRAP_FREE_MEMBER_FUNC_2(fPtr, T1, T2, T3) \
  +[](const Tensor& self, T1 v1, T2 v2, T3 v3) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((self.*fPtr)(v1, v2, v3)); \
  }

  #define WRAP_FREE_FUNC_1(fPtr, T1) \
  +[](T1 v1) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((*fPtr)(v1)); \
  }

  #define WRAP_FREE_FUNC_2(fPtr, T1, T2) \
  +[](T1 v1, T2 v2) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((*fPtr)(v1, v2)); \
  }

  #define WRAP_FREE_FUNC_3(fPtr, T1, T2, T3) \
  +[](T1 v1, T2 v2, T3 v3) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((*fPtr)(v1, v2, v3)); \
  }

  #define WRAP_FREE_FUNC_4(fPtr, T) \
  +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
    return (*fPtr)(self.getSharedPtr(), val); \
  }

  #define WRAP_FREE_FUNC_5(fPtr) \
  +[](const Tensor& self, const Tensor& other) -> std::shared_ptr<Tensor> { \
    return (*fPtr)(self.getSharedPtr(), other.getSharedPtr()); \
  }

  #define WRAP_FREE_FUNC_6(fPtr, T) \
  +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
    return (*fPtr)(val, self.getSharedPtr()); \
  }

  #define WRAP_FREE_FUNC_7(fPtr) \
  +[](const Tensor& self) -> std::shared_ptr<Tensor> { \
    return (*fPtr)(self.getSharedPtr()); \
  }

  #define WRAP_FREE_FUNC_8(fPtr, T1, T2, T3, T4) \
  +[](T1 v1, T2 v2, T3 v3, T4 v4) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((*fPtr)(v1, v2, v3, v4)); \
  }

  #define WRAP_FREE_FUNC_9(fPtr, T1, T2, T3, T4, T5) \
  +[](T1 v1, T2 v2, T3 v3, T4 v4, T5 v5) -> std::shared_ptr<Tensor> { \
    return std::make_shared<Tensor>((*fPtr)(v1, v2, v3, v4, v5)); \
  }

  #define WRAP_FUNC_AND_CONVERT_DTYPE_1(method) \
  +[](const Tensor& self, int v1) -> ftype { \
    return self.method(static_cast<tensorSize_t>(v1)); \
  }

  #define WRAP_FUNC_AND_CONVERT_DTYPE_2(method) \
  +[](const Tensor& self, int v1, int v2) -> ftype { \
    return self.method(static_cast<tensorDim_t>(v1), static_cast<tensorDim_t>(v2)); \
  }

  #define WRAP_FUNC_AND_CONVERT_DTYPE_3(method) \
  +[](const Tensor& self, int v1, int v2, int v3) -> ftype { \
    return self.method(static_cast<tensorDim_t>(v1), static_cast<tensorDim_t>(v2), \
                       static_cast<tensorDim_t>(v3)); \
  }

  #define WRAP_FUNC_AND_CONVERT_DTYPE_4(method) \
  +[](const Tensor& self, int v1, int v2, int v3, int v4) -> ftype { \
    return self.method(static_cast<tensorDim_t>(v1), static_cast<tensorDim_t>(v2), \
                       static_cast<tensorDim_t>(v3), static_cast<tensorDim_t>(v4)); \
  }

  // classes
  class_<Dimension>("Dimension", no_init)
    .add_property("list", &Dimension::get)
    .def("__str__", &Py_Util::toString<Dimension>)
    .def("__eq__", Py_DataModeling::dimEquals1)
    .def("__eq__", Py_DataModeling::dimEquals2)
    .def("__ne__", Py_DataModeling::nDimEquals1)
    .def("__ne__", Py_DataModeling::nDimEquals2)
  ;

  enum_<Device>("Device")
    .value("CPU", Device::CPU)
    .value("CUDA", Device::CUDA)
  ;

  // register implicit dtype conversion
  custom_converters::PyListToVectorConverter<tensorDim_t>();
  custom_converters::PyListToVectorConverter<ftype>();

  // to convert std::shared_ptr<const Tensor> to std::shared_ptr<Tensor>> in Python
  boost::python::register_ptr_to_python< std::shared_ptr<const Tensor> >();

  // we manage via shared_ptr, since we deleted copy-ctor
  class_<Tensor, std::shared_ptr<Tensor>, boost::noncopyable>("Tensor", no_init)
    .def(init<const std::vector<tensorDim_t>&, optional<bool> >())
    .def(init<const std::vector<tensorDim_t>&, Device, optional<bool> >())
    .def(init<const std::vector<tensorDim_t>&, const std::vector<ftype>&, optional<bool> >())
    .def(init<const std::vector<tensorDim_t>&, const std::vector<ftype>&, Device, optional<bool> >())
        
    // static creation methods
    .def("ones", WRAP_FREE_FUNC_1(Py_DataModeling::Ones0, std::vector<tensorDim_t>))
    .def("ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones1, std::vector<tensorDim_t>, Device))
    .def("ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones2, std::vector<tensorDim_t>, const bool))
    .def("ones", WRAP_FREE_FUNC_3(Py_DataModeling::Ones3, std::vector<tensorDim_t>, Device, const bool))
    .staticmethod("ones")

    .def("zeros", WRAP_FREE_FUNC_1(Py_DataModeling::Zeros0, std::vector<tensorDim_t>))
    .def("zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros1, std::vector<tensorDim_t>, Device))
    .def("zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros2, std::vector<tensorDim_t>, const bool))
    .def("zeros", WRAP_FREE_FUNC_3(Py_DataModeling::Zeros3, std::vector<tensorDim_t>, Device, const bool))
    .staticmethod("zeros")

    .def("gauss", WRAP_FREE_FUNC_2(Py_DataModeling::Gaussian0, std::vector<tensorDim_t>, ftype))
    .def("gauss", WRAP_FREE_FUNC_3(Py_DataModeling::Gaussian1, std::vector<tensorDim_t>, Device, ftype))
    .def("gauss", WRAP_FREE_FUNC_3(Py_DataModeling::Gaussian2, std::vector<tensorDim_t>, ftype, const bool))
    .def("gauss", WRAP_FREE_FUNC_8(Py_DataModeling::Gaussian3, std::vector<tensorDim_t>, Device, ftype, const bool))
    .staticmethod("gauss")

    // properties
    .add_property("device", &Tensor::getDevice, &Tensor::setDevice)
    .add_property("dims", make_function(&Tensor::getDims, return_internal_reference<>()))
    .add_property("grads", &Tensor::getGrads)
    .add_property("requiresGrad", &Tensor::getRequiresGrad, &Tensor::setRequiresGrad)
    .add_property("size", &Tensor::getSize)

    // operators
    .def("__str__", &Py_Util::toString<Tensor>)
    .def("__repr__", &Py_Util::toString<Tensor>)
    .def("__len__", &Tensor::getSize)
    .def("__getitem__", WRAP_FREE_FUNC_4(&Py_DataModeling::getItemAsTensor1, tensorSize_t))
    .def("__getitem__", WRAP_FREE_FUNC_4(&Py_DataModeling::getItemAsTensor2, std::vector<tensorDim_t>))
    .def("__setitem__", &Py_DataModeling::tensorSetItem)

    // arithmetics
    .def("__matmul__", WRAP_FREE_FUNC_5(Py_DataModeling::matmul))
    .def("__add__", WRAP_FREE_FUNC_5(Py_DataModeling::elementwiseadd)) // elementwise add
    .def("__add__", WRAP_FREE_FUNC_4(Py_DataModeling::scalaradd, ftype))
    .def("__radd__", WRAP_FREE_FUNC_6(Py_DataModeling::rscalaradd, ftype))

    .def("__mul__", WRAP_FREE_FUNC_5(Py_DataModeling::elementwisemul)) // elementwise mult
    .def("__mul__", WRAP_FREE_FUNC_4(Py_DataModeling::scalarmul, ftype))
    .def("__rmul__", WRAP_FREE_FUNC_6(Py_DataModeling::rscalarmul, ftype))
        
    .def("__sub__", WRAP_FREE_FUNC_4(Py_DataModeling::scalarsub, ftype))
    .def("__truediv__", WRAP_FREE_FUNC_4(Py_DataModeling::scalardiv, ftype))

    // member functions
    .def("getitem", WRAP_FUNC_AND_CONVERT_DTYPE_1(Tensor::get))
    .def("getitem", WRAP_FUNC_AND_CONVERT_DTYPE_2(Tensor::get))
    .def("getitem", WRAP_FUNC_AND_CONVERT_DTYPE_3(Tensor::get))
    .def("getitem", WRAP_FUNC_AND_CONVERT_DTYPE_4(Tensor::get))
    .def("getitem", Py_DataModeling::getItemVector) // the vector arg

    .def("sum", WRAP_FREE_FUNC_7(&(cgraph::sumTensor)))
        
    .def("reset", Py_DataModeling::reset1)
    .def("reset", Py_DataModeling::reset2)
    
    .def("hasGrads", &Tensor::hasGrads)
    .def("hasGrads", +[](const std::shared_ptr<Tensor>& t) -> bool {
      return t->hasGrads();
      })

    .def("transpose", WRAP_FREE_MEMBER_FUNC_1(Py_DataModeling::transpose1, int, int))
    .def("transpose", WRAP_FREE_MEMBER_FUNC_2(Py_DataModeling::transpose2, int, int, bool))
    .def("transposeThis", Py_DataModeling::transposeThis1)
    .def("transposeThis", Py_DataModeling::transposeThis2)
        
    .def("backward", &Tensor::backward)
  ;

  // free functions
  def("Ones", WRAP_FREE_FUNC_1(Py_DataModeling::Ones0, std::vector<tensorDim_t>));
  def("Ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones1, std::vector<tensorDim_t>, Device));
  def("Ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones2, std::vector<tensorDim_t>, const bool));
  def("Ones", WRAP_FREE_FUNC_3(Py_DataModeling::Ones3, std::vector<tensorDim_t>, Device, const bool));

  def("Zeros", WRAP_FREE_FUNC_1(Py_DataModeling::Zeros0, std::vector<tensorDim_t>));
  def("Zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros1, std::vector<tensorDim_t>, Device));
  def("Zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros2, std::vector<tensorDim_t>, const bool));
  def("Zeros", WRAP_FREE_FUNC_3(Py_DataModeling::Zeros3, std::vector<tensorDim_t>, Device, const bool));

  def("Gaussian", WRAP_FREE_FUNC_2(Py_DataModeling::Gaussian0, std::vector<tensorDim_t>, ftype));
  def("Gaussian", WRAP_FREE_FUNC_3(Py_DataModeling::Gaussian1, std::vector<tensorDim_t>, Device, ftype));
  def("Gaussian", WRAP_FREE_FUNC_3(Py_DataModeling::Gaussian2, std::vector<tensorDim_t>, ftype, const bool));
  def("Gaussian", WRAP_FREE_FUNC_8(Py_DataModeling::Gaussian3, std::vector<tensorDim_t>, Device, ftype, const bool));
}