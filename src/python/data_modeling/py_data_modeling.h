/**
 * @file tensor.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"

#include "python_templates.h"
#include "custom_converters.h"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/object.hpp>

namespace Py_DataModeling {
    ftype tensorGetItem(const Tensor& self, boost::python::object index);
    void tensorSetItem(Tensor& self, boost::python::object index, ftype value);

    // need wrappers for default arguments, see
    // https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/functions.html
    auto OnesWrapper0(std::vector<tensorDim_t> dims) { 
        return TensorFunctions::Ones(std::move(dims)); 
    }

    auto OnesWrapper1(std::vector<tensorDim_t> dims, Device d) { 
        return TensorFunctions::Ones(std::move(dims), d); 
    }

    auto ZerosWrapper0(std::vector<tensorDim_t> dims) { 
        return TensorFunctions::Zeros(std::move(dims)); 
    }

    auto ZerosWrapper1(std::vector<tensorDim_t> dims, Device d) { 
        return TensorFunctions::Zeros(std::move(dims), d); 
    }

    auto GaussianWrapper0(std::vector<tensorDim_t> dims) { 
        return TensorFunctions::Gaussian(std::move(dims)); 
    }

    auto GaussianWrapper1(std::vector<tensorDim_t> dims, Device d) { 
        return TensorFunctions::Gaussian(std::move(dims), d); 
    }

    Tensor    (*Ones0)(std::vector<tensorDim_t>)                         = &OnesWrapper0;
    Tensor    (*Ones1)(std::vector<tensorDim_t>, Device)                 = &OnesWrapper1;
    Tensor    (*Ones2)(std::vector<tensorDim_t>, const bool)             = &(TensorFunctions::Ones);
    Tensor    (*Ones3)(std::vector<tensorDim_t>, Device, const bool)     = &(TensorFunctions::Ones);

    Tensor    (*Zeros0)(std::vector<tensorDim_t>)                        = &ZerosWrapper0;
    Tensor    (*Zeros1)(std::vector<tensorDim_t>, Device)                = &ZerosWrapper1;
    Tensor    (*Zeros2)(std::vector<tensorDim_t>, const bool)            = &(TensorFunctions::Zeros);
    Tensor    (*Zeros3)(std::vector<tensorDim_t>, Device, const bool)    = &(TensorFunctions::Zeros);

    Tensor    (*Gaussian0)(std::vector<tensorDim_t>)                     = &GaussianWrapper0;
    Tensor    (*Gaussian1)(std::vector<tensorDim_t>, Device)             = &GaussianWrapper1;
    Tensor    (*Gaussian2)(std::vector<tensorDim_t>, const bool)         = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian3)(std::vector<tensorDim_t>, Device, const bool) = &(TensorFunctions::Gaussian);

    void    (Tensor::*reset1)(const ftype)                                = &Tensor::reset;
    void    (Tensor::*reset2)(const utility::InitClass)                   = &Tensor::reset;

    void    (Tensor::*transposeThis1)()                                   = &Tensor::transposeThis;
    void    (Tensor::*transposeThis2)(int, int)                           = &Tensor::transposeThis;
    Tensor  (Tensor::*transpose1)(int, int) const                         = &Tensor::transpose;
    Tensor  (Tensor::*transpose2)(int, int, bool) const                   = &Tensor::transpose;
}

BOOST_PYTHON_MODULE(py_data_modeling)
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

    // register implicit dtype conversion
    converters::PyListToVectorConverter<tensorDim_t>();

    // classes
    class_<Dimension>("Dimension", no_init)
        .add_property("list", &Dimension::get)
        .def("__str__", &Py_Util::toString<Dimension>)
    ;

    enum_<Device>("Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
    ;

    // we manage via shared_ptr, since we deleted copy-ctor
    class_<Tensor, std::shared_ptr<Tensor>, boost::noncopyable>("Tensor", no_init)
        .def(init<const std::vector<tensorDim_t>&, optional<bool> >())
        .def(init<const std::vector<tensorDim_t>&, optional<Device, bool> >())
        .add_property("device", &Tensor::getDevice, &Tensor::setDevice)
        .add_property("dims", make_function(&Tensor::getDims, return_internal_reference<>()))
        .add_property("grads", make_function(&Tensor::getGrads, return_internal_reference<>()))
        .def("__str__", &Py_Util::toString<Tensor>)
        .def("__repr__", &Py_Util::toString<Tensor>)
        .def("__getitem__", &Py_DataModeling::tensorGetItem)
        .def("__setitem__", &Py_DataModeling::tensorSetItem)
        .def("__matmul__", WRAP_TENSOR_METHOD_1(matmul))
        .def("__add__", WRAP_TENSOR_METHOD_1(operator+)) // elementwise add
        .def("__mul__", WRAP_TENSOR_METHOD_1(operator*)) // elementwise mult
        .def("__mul__", WRAP_SCALAR(operator*, float))
        .def("__rmul__", WRAP_SCALAR_REVERSE(*, float))
        .def("__add__", WRAP_SCALAR(operator+, float))
        .def("__radd__", WRAP_SCALAR_REVERSE(+, float))
        .def("__sub__", WRAP_SCALAR(operator-, float))
        .def("__truediv__", WRAP_SCALAR(operator/, float))
        .def("reset", Py_DataModeling::reset1)
        .def("reset", Py_DataModeling::reset2)
        .def("transpose", WRAP_FREE_MEMBER_FUNC_1(Py_DataModeling::transpose1, int, int))
        .def("transpose", WRAP_FREE_MEMBER_FUNC_2(Py_DataModeling::transpose2, int, int, bool))
        .def("transposeThis", Py_DataModeling::transposeThis1)
        .def("transposeThis", Py_DataModeling::transposeThis2)
        .def("backward", &Tensor::backward)
    ;

    // functions
    def("Ones", WRAP_FREE_FUNC_1(Py_DataModeling::Ones0, std::vector<tensorDim_t>));
    def("Ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones1, std::vector<tensorDim_t>, Device));
    def("Ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones2, std::vector<tensorDim_t>, const bool));
    def("Ones", WRAP_FREE_FUNC_3(Py_DataModeling::Ones3, std::vector<tensorDim_t>, Device, const bool));

    def("Zeros", WRAP_FREE_FUNC_1(Py_DataModeling::Zeros0, std::vector<tensorDim_t>));
    def("Zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros1, std::vector<tensorDim_t>, Device));
    def("Zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros2, std::vector<tensorDim_t>, const bool));
    def("Zeros", WRAP_FREE_FUNC_3(Py_DataModeling::Zeros3, std::vector<tensorDim_t>, Device, const bool));

    def("Gaussian", WRAP_FREE_FUNC_1(Py_DataModeling::Gaussian0, std::vector<tensorDim_t>));
    def("Gaussian", WRAP_FREE_FUNC_2(Py_DataModeling::Gaussian1, std::vector<tensorDim_t>, Device));
    def("Gaussian", WRAP_FREE_FUNC_2(Py_DataModeling::Gaussian2, std::vector<tensorDim_t>, const bool));
    def("Gaussian", WRAP_FREE_FUNC_3(Py_DataModeling::Gaussian3, std::vector<tensorDim_t>, Device, const bool));
}