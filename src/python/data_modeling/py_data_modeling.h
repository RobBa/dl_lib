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
#include "python_templates.h"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/object.hpp>

namespace Py_DataModeling {
    ftype tensorGetItem(const Tensor& self, boost::python::object index);
    ftype tensorSetItem(Tensor& self, boost::python::object index, ftype value);
}

BOOST_PYTHON_MODULE(py_data_modeling)
{
    using namespace boost::python;

    class_<Dimension>("Dimension", no_init)
        .def("get", &Dimension::get)
        .def("getTotalSize", &Dimension::getTotalSize)
        .def("__str__", &toString<Dimension>)
    ;

    enum_<Device>("Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
    ;

    class_<Tensor>("Tensor", init< optional<Device> >())
        .def(init<tensorDim_t, optional<Device> >())
        .def(init<tensorDim_t, tensorDim_t, optional<Device> >())
        .def(init<tensorDim_t, tensorDim_t, tensorDim_t, optional<Device> >())
        .def(init<tensorDim_t, tensorDim_t, tensorDim_t, tensorDim_t, optional<Device> >())
        .def("dim", &Tensor::getDims, return_internal_reference<>())
        .def(self * self)
        .def("__str__", &toString<Tensor>)
        .def("__getitem__", &Py_DataModeling::tensorGetItem)
        .def("__setitem__", &Py_DataModeling::tensorSetItem)
    ;

}