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

    class_<LayerBaseWrap, boost::noncopyable>("LayerBase", no_init)
        .def("forward", pure_virtual(&layers::LayerBase::forward))
        .def("getDims", &layers::LayerBase::getDims, return_internal_reference<>())
        //.def("backward", &FfLayer::backward)
    ;

    class_<layers::FfLayer, bases<layers::LayerBase> >("FfLayer", init<std::uint16_t, std::uint16_t>())
        .def("forward", &layers::FfLayer::forward)
        //.def("backward", &FfLayer::backward)
    ;
}