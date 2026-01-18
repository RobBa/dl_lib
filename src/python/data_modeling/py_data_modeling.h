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

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>

BOOST_PYTHON_MODULE(py_data_modeling)
{
    using namespace boost::python;

    class_<Dimension>("Dimension", no_init)
        .def("get", &Dimension::get)
        .def("getTotalSize", &Dimension::getTotalSize)
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
        .def("getDims", &Tensor::getDims, return_internal_reference<>())
        .def(self * self)
    ;

}