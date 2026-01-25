/**
 * @file layers.h
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
#include <boost/python/wrapper.hpp>
#include <boost/python/return_internal_reference.hpp>

BOOST_PYTHON_MODULE(py_layers)
{
    using namespace boost::python;

    /**
     * @brief Wrapper class needed for Boost Python to get the virtual function working 
     * the way it is intended. See documentation here: 
     * https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/exposing.html
     * 
     */
    struct LayerBaseWrap : layers::LayerBase, wrapper<layers::LayerBase> {
        Tensor forward(const Tensor& input) const
        {
            return this->get_override("forward")(input);
        }
    };

    /**
     * @brief We need these pointers to wrap overloading
     */
    ftype (layers::LayerBase::*get0)(void)               const = &layers::LayerBase::get;
    ftype (layers::LayerBase::*get1)(int)                const = &layers::LayerBase::get;
    ftype (layers::LayerBase::*get2)(int, int)           const = &layers::LayerBase::get;
    ftype (layers::LayerBase::*get3)(int, int, int)      const = &layers::LayerBase::get;
    ftype (layers::LayerBase::*get4)(int, int, int, int) const = &layers::LayerBase::get;

    void (layers::LayerBase::*set0)(ftype)                     = &layers::LayerBase::set;
    void (layers::LayerBase::*set1)(ftype, int)                = &layers::LayerBase::set;
    void (layers::LayerBase::*set2)(ftype, int, int)           = &layers::LayerBase::set;
    void (layers::LayerBase::*set3)(ftype, int, int, int)      = &layers::LayerBase::set;
    void (layers::LayerBase::*set4)(ftype, int, int, int, int) = &layers::LayerBase::set;

    class_<LayerBaseWrap, boost::noncopyable>("LayerBase", no_init)
        .def("forward", pure_virtual(&layers::LayerBase::forward))
        //.def("backward", &FfLayer::backward)
        .def("getDims", &layers::LayerBase::getDims, return_internal_reference<>())
        .def("getTensor", &layers::LayerBase::getDims, return_internal_reference<>())
        .def("__getitem__", get0)
        .def("__getitem__", get1)
        .def("__getitem__", get2)
        .def("__getitem__", get3)
        .def("__getitem__", get4)
        .def("__setitem__", set0)
        .def("__setitem__", set1)
        .def("__setitem__", set2)
        .def("__setitem__", set3)
        .def("__setitem__", set4)
    ;

    class_<layers::FfLayer, bases<layers::LayerBase> >("FfLayer", init<tensorDim_t, tensorDim_t>())
        .def("forward", &layers::FfLayer::forward)
        //.def("backward", &FfLayer::backward)
    ;
}