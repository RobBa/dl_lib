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

#include "tensor.h"
#include "dim_type.h"
#include "tensor_functions.h"

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

    std::shared_ptr<Tensor>    (*Ones0)(std::vector<tensorDim_t>)                         = &OnesWrapper0;
    std::shared_ptr<Tensor>    (*Ones1)(std::vector<tensorDim_t>, Device)                 = &OnesWrapper1;
    std::shared_ptr<Tensor>    (*Ones2)(std::vector<tensorDim_t>, const bool)             = &(TensorFunctions::Ones);
    std::shared_ptr<Tensor>    (*Ones3)(std::vector<tensorDim_t>, Device, const bool)     = &(TensorFunctions::Ones);

    std::shared_ptr<Tensor>    (*Zeros0)(std::vector<tensorDim_t>)                        = &ZerosWrapper0;
    std::shared_ptr<Tensor>    (*Zeros1)(std::vector<tensorDim_t>, Device)                = &ZerosWrapper1;
    std::shared_ptr<Tensor>    (*Zeros2)(std::vector<tensorDim_t>, const bool)            = &(TensorFunctions::Zeros);
    std::shared_ptr<Tensor>    (*Zeros3)(std::vector<tensorDim_t>, Device, const bool)    = &(TensorFunctions::Zeros);

    std::shared_ptr<Tensor>    (*Gaussian0)(std::vector<tensorDim_t>)                     = &GaussianWrapper0;
    std::shared_ptr<Tensor>    (*Gaussian1)(std::vector<tensorDim_t>, Device)             = &GaussianWrapper1;
    std::shared_ptr<Tensor>    (*Gaussian2)(std::vector<tensorDim_t>, const bool)         = &(TensorFunctions::Gaussian);
    std::shared_ptr<Tensor>    (*Gaussian3)(std::vector<tensorDim_t>, Device, const bool) = &(TensorFunctions::Gaussian);

    void    (Tensor::*reset1)(const ftype)                         = &Tensor::reset;
    void    (Tensor::*reset2)(const utility::InitClass)            = &Tensor::reset;

    template<typename Func>
    auto WrapReturnedTensor(Func f) {
        return [f](const Tensor& self, auto&&... args) -> std::shared_ptr<Tensor> {
            return std::make_shared<Tensor>(f(self, std::forward<decltype(args)>(args)...));
        };
    }
}

BOOST_PYTHON_MODULE(py_data_modeling)
{
    using namespace boost::python;

    converters::PyListToVectorConverter<tensorDim_t>();

    // classes
    class_<Dimension>("Dimension", no_init)
        .add_property("list", &Dimension::get)
        .def("__str__", &toString<Dimension>)
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
        .def("__str__", &toString<Tensor>)
        .def("__repr__", &toString<Tensor>)
        .def("__getitem__", &Py_DataModeling::tensorGetItem)
        .def("__setitem__", &Py_DataModeling::tensorSetItem)
        .def("__matmul__", +[](const Tensor& self, const Tensor& other) -> std::shared_ptr<Tensor> {
            return std::make_shared<Tensor>(self.matmul(other));})
        .def(self + self)
        .def(self * self)
        .def("reset", Py_DataModeling::reset1)
        .def("reset", Py_DataModeling::reset2)
    ;

    // functions
    def("Ones", Py_DataModeling::Ones0);
    def("Ones", Py_DataModeling::Ones1);
    def("Ones", Py_DataModeling::Ones2);
    def("Ones", Py_DataModeling::Ones3);

    def("Zeros", Py_DataModeling::Zeros0);
    def("Zeros", Py_DataModeling::Zeros1);
    def("Zeros", Py_DataModeling::Zeros2);
    def("Zeros", Py_DataModeling::Zeros3);

    def("Gaussian", Py_DataModeling::Gaussian0);
    def("Gaussian", Py_DataModeling::Gaussian1);
    def("Gaussian", Py_DataModeling::Gaussian2);
    def("Gaussian", Py_DataModeling::Gaussian3);
}