/**
 * @file layers_cpu.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-11-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ff_layer_cpu.h"

#include <boost/python.hpp>

/* BOOST_PYTHON_MODULE(layers_cpu)
{
    using namespace boost::python;

    class_<ff_layer_cpu>("ff_layer_cpu", init<int, int>())
        .def("forward", &ff_layer_cpu::forward)
        //.def("backward", &ff_layer_cpu::backward)
    ;
} */