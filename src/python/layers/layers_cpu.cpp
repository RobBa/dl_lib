
//#include "ff_layer_cpu.h"

#include <boost/python.hpp>
//#include <boost/python/detail/wrap_python.hpp>

#include "ff_layer_cpu.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(layers_cpu)
{
    Py_Initialize();
    class_<ff_layer_cpu>("ff_layer_cpu", init<int, int>())
        .def("forward", &ff_layer_cpu::forward)
        //.def("backward", &ff_layer_cpu::backward)
    ;
}