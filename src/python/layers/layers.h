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

#include "ff_layer.h"

#include <boost/python.hpp>

BOOST_PYTHON_MODULE(layers_cpu)
{
    using namespace boost::python;

    class_<layers::FfLayer>("FfLayer", init<int, int>())
        .def("forward", &FfLayer::forward)
        //.def("backward", &FfLayer::backward)
    ;
}