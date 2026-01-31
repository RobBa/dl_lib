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

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/object.hpp>

namespace Py_DataModeling {
    ftype tensorGetItem(const Tensor& self, boost::python::object index);
    void tensorSetItem(Tensor& self, boost::python::object index, ftype value);

    Tensor    (*Ones00)()                              = &(TensorFunctions::Ones);
    Tensor    (*Ones01)(int)                           = &(TensorFunctions::Ones);
    Tensor    (*Ones02)(int, int)                      = &(TensorFunctions::Ones);
    Tensor    (*Ones03)(int, int, int)                 = &(TensorFunctions::Ones);
    Tensor    (*Ones04)(int, int, int, int)            = &(TensorFunctions::Ones);

    Tensor    (*Ones10)(Device)                        = &(TensorFunctions::Ones);
    Tensor    (*Ones11)(Device, int)                   = &(TensorFunctions::Ones);
    Tensor    (*Ones12)(Device, int, int)              = &(TensorFunctions::Ones);
    Tensor    (*Ones13)(Device, int, int, int)         = &(TensorFunctions::Ones);
    Tensor    (*Ones14)(Device, int, int, int, int)    = &(TensorFunctions::Ones);

    Tensor    (*Zeros00)()                              = &(TensorFunctions::Zeros);
    Tensor    (*Zeros01)(int)                           = &(TensorFunctions::Zeros);
    Tensor    (*Zeros02)(int, int)                      = &(TensorFunctions::Zeros);
    Tensor    (*Zeros03)(int, int, int)                 = &(TensorFunctions::Zeros);
    Tensor    (*Zeros04)(int, int, int, int)            = &(TensorFunctions::Zeros);

    Tensor    (*Zeros10)(Device)                        = &(TensorFunctions::Zeros);
    Tensor    (*Zeros11)(Device, int)                   = &(TensorFunctions::Zeros);
    Tensor    (*Zeros12)(Device, int, int)              = &(TensorFunctions::Zeros);
    Tensor    (*Zeros13)(Device, int, int, int)         = &(TensorFunctions::Zeros);
    Tensor    (*Zeros14)(Device, int, int, int, int)    = &(TensorFunctions::Zeros);

    Tensor    (*Gaussian00)()                              = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian01)(int)                           = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian02)(int, int)                      = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian03)(int, int, int)                 = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian04)(int, int, int, int)            = &(TensorFunctions::Gaussian);

    Tensor    (*Gaussian10)(Device)                        = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian11)(Device, int)                   = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian12)(Device, int, int)              = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian13)(Device, int, int, int)         = &(TensorFunctions::Gaussian);
    Tensor    (*Gaussian14)(Device, int, int, int, int)    = &(TensorFunctions::Gaussian);

    void    (Tensor::*reset1)(const ftype)                         = &Tensor::reset;
    void    (Tensor::*reset2)(const utility::InitClass)            = &Tensor::reset;
}

BOOST_PYTHON_MODULE(py_data_modeling)
{
    using namespace boost::python;

    // classes
    class_<Dimension>("Dimension", no_init)
        .def("get", &Dimension::get)
        .def("__str__", &toString<Dimension>)
    ;

    enum_<Device>("Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
    ;

    class_<Tensor>("Tensor")
        .def(init<tensorDim_t>())
        .def(init<tensorDim_t, tensorDim_t >())
        .def(init<tensorDim_t, tensorDim_t, tensorDim_t >())
        .def(init<tensorDim_t, tensorDim_t, tensorDim_t, tensorDim_t >())
        .def(init<Device, tensorDim_t>())
        .def(init<Device, tensorDim_t, tensorDim_t >())
        .def(init<Device, tensorDim_t, tensorDim_t, tensorDim_t >())
        .def(init<Device, tensorDim_t, tensorDim_t, tensorDim_t, tensorDim_t >())
        .def("__str__", &toString<Tensor>)
        .def("__getitem__", &Py_DataModeling::tensorGetItem)
        .def("__setitem__", &Py_DataModeling::tensorSetItem)
        .def("dim", &Tensor::getDims, return_internal_reference<>())
        .def(self * self)
        .def("reset", Py_DataModeling::reset1)
        .def("reset", Py_DataModeling::reset2)
        .def("setDevice", &Tensor::setDevice)
        .def("getDevice", &Tensor::getDevice)
    ;

    // functions
    def("Ones", Py_DataModeling::Ones00);
    def("Ones", Py_DataModeling::Ones01);
    def("Ones", Py_DataModeling::Ones02);
    def("Ones", Py_DataModeling::Ones03);
    def("Ones", Py_DataModeling::Ones04);

    def("Ones", Py_DataModeling::Ones10);
    def("Ones", Py_DataModeling::Ones11);
    def("Ones", Py_DataModeling::Ones12);
    def("Ones", Py_DataModeling::Ones13);
    def("Ones", Py_DataModeling::Ones14);

    def("Zeros", Py_DataModeling::Zeros00);
    def("Zeros", Py_DataModeling::Zeros01);
    def("Zeros", Py_DataModeling::Zeros02);
    def("Zeros", Py_DataModeling::Zeros03);
    def("Zeros", Py_DataModeling::Zeros04);

    def("Zeros", Py_DataModeling::Zeros10);
    def("Zeros", Py_DataModeling::Zeros11);
    def("Zeros", Py_DataModeling::Zeros12);
    def("Zeros", Py_DataModeling::Zeros13);
    def("Zeros", Py_DataModeling::Zeros14);

    def("Gaussian", Py_DataModeling::Gaussian00);
    def("Gaussian", Py_DataModeling::Gaussian01);
    def("Gaussian", Py_DataModeling::Gaussian02);
    def("Gaussian", Py_DataModeling::Gaussian03);
    def("Gaussian", Py_DataModeling::Gaussian04);

    def("Gaussian", Py_DataModeling::Gaussian10);
    def("Gaussian", Py_DataModeling::Gaussian11);
    def("Gaussian", Py_DataModeling::Gaussian12);
    def("Gaussian", Py_DataModeling::Gaussian13);
    def("Gaussian", Py_DataModeling::Gaussian14);
}