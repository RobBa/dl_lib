/**
 * @file py_sys.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */


#include "system/sys_functions.h"

#include <boost/python.hpp>

BOOST_PYTHON_MODULE(_sys)
{
    using namespace boost::python;
    
    def("setGlobalDevice", &sys::setGlobalDevice);
    def("getGlobalDevice", &sys::getGlobalDevice);
}