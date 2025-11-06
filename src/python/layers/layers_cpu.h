#pragma once

#include <boost/python.hpp>

#include "ff_layer_cpu.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(PythonLib)
{
    class_<ff_layer_cpu>("ff_layer_cpu", init<int, int>())
        .def("forward", &ff_layer_cpu::forward)
        //.def("backward", &ff_layer_cpu::backward)
    ;
}