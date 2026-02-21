/**
 * @file py_data_modeling.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-21
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "data_modeling/tensor.h"

#include "py_data_modeling_util.h"
#include "python_templates.h"
#include "custom_converters.h"

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"
#include "computational_graph/graph_creation.h"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>

BOOST_PYTHON_MODULE(py_data_modeling)
{
    using namespace boost::python;

    // some macros to make code below easier to read
    #define WRAP_TENSOR_METHOD_1(method) \
    +[](const Tensor& self, const Tensor& other) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>(self.method(other)); \
    }

    #define WRAP_SCALAR(method, T) \
    +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>(self.method(val)); \
    }

    #define WRAP_SCALAR_REVERSE(op, T) \
    +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>(val op self); \
    }

    // different, since those are not methods anymore
    #define WRAP_FREE_MEMBER_FUNC_1(fPtr, T1, T2) \
    +[](const Tensor& self, int v1, int v2) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>((self.*fPtr)(v1, v2)); \
    }

    #define WRAP_FREE_MEMBER_FUNC_2(fPtr, T1, T2, T3) \
    +[](const Tensor& self, T1 v1, T2 v2, T3 v3) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>((self.*fPtr)(v1, v2, v3)); \
    }

    #define WRAP_FREE_FUNC_1(fPtr, T1) \
    +[](T1 v1) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>((*fPtr)(v1)); \
    }

    #define WRAP_FREE_FUNC_2(fPtr, T1, T2) \
    +[](T1 v1, T2 v2) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>((*fPtr)(v1, v2)); \
    }

    #define WRAP_FREE_FUNC_3(fPtr, T1, T2, T3) \
    +[](T1 v1, T2 v2, T3 v3) -> std::shared_ptr<Tensor> { \
        return std::make_shared<Tensor>((*fPtr)(v1, v2, v3)); \
    }

    #define WRAP_FREE_FUNC_4(fPtr, T) \
    +[](const Tensor& self, T val) -> std::shared_ptr<Tensor> { \
        return (*fPtr)(self.getSharedPtr(), val); \
    }

    #define WRAP_FREE_FUNC_5(fPtr) \
    +[](const Tensor& self, const Tensor& other) -> std::shared_ptr<Tensor> { \
        return (*fPtr)(self.getSharedPtr(), other.getSharedPtr()); \
    }

    // classes
    class_<Dimension>("Dimension", no_init)
        .add_property("list", &Dimension::getItem)
        .def("__str__", &Py_Util::toString<Dimension>)
    ;

    enum_<Device>("Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
    ;

    // register implicit dtype conversion
    converters::PyListToVectorConverter<tensorDim_t>();
    converters::PyListToVectorConverter<ftype>();

    // we manage via shared_ptr, since we deleted copy-ctor
    class_<Tensor, std::shared_ptr<Tensor>, boost::noncopyable>("Tensor", no_init)
        .def(init<const std::vector<tensorDim_t>&, optional<bool> >())
        .def(init<const std::vector<tensorDim_t>&, Device, optional<bool> >())
        .def(init<const std::vector<tensorDim_t>&, const std::vector<ftype>&, optional<bool> >())
        .def(init<const std::vector<tensorDim_t>&, const std::vector<ftype>&, Device, optional<bool> >())
        .add_property("device", &Tensor::getDevice, &Tensor::setDevice)
        .add_property("dims", make_function(&Tensor::getDims, return_internal_reference<>()))
        .add_property("grads", make_function(&Tensor::getGrads, return_internal_reference<>()))
        .def("__str__", &Py_Util::toString<Tensor>)
        .def("__repr__", &Py_Util::toString<Tensor>)
        .def("__getitem__", WRAP_FREE_FUNC_4(&Py_DataModeling::getItemAsTensor1, tensorSize_t))
        .def("__getitem__", WRAP_FREE_FUNC_4(&Py_DataModeling::getItemAsTensor2, std::vector<tensorDim_t>))
        .def("__setitem__", &Py_DataModeling::tensorSetItem)
        .def("getvalue", &Py_DataModeling::tensorGetItem)
        .def("__matmul__", WRAP_TENSOR_METHOD_1(matmul))
        .def("__add__", WRAP_TENSOR_METHOD_1(operator+)) // elementwise add
        .def("__mul__", WRAP_FREE_FUNC_5(&Py_DataModeling::elementwisemul)) // elementwise mult
        .def("__mul__", WRAP_SCALAR(operator*, float))
        .def("__rmul__", WRAP_SCALAR_REVERSE(*, float))
        .def("__add__", WRAP_SCALAR(operator+, float))
        .def("__radd__", WRAP_SCALAR_REVERSE(+, float))
        .def("__sub__", WRAP_SCALAR(operator-, float))
        .def("__truediv__", WRAP_SCALAR(operator/, float))
        .def("reset", Py_DataModeling::reset1)
        .def("reset", Py_DataModeling::reset2)
        .def("transpose", WRAP_FREE_MEMBER_FUNC_1(Py_DataModeling::transpose1, int, int))
        .def("transpose", WRAP_FREE_MEMBER_FUNC_2(Py_DataModeling::transpose2, int, int, bool))
        .def("transposeThis", Py_DataModeling::transposeThis1)
        .def("transposeThis", Py_DataModeling::transposeThis2)
        .def("backward", &Tensor::backward)
    ;

    // functions
    def("Ones", WRAP_FREE_FUNC_1(Py_DataModeling::Ones0, std::vector<tensorDim_t>));
    def("Ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones1, std::vector<tensorDim_t>, Device));
    def("Ones", WRAP_FREE_FUNC_2(Py_DataModeling::Ones2, std::vector<tensorDim_t>, const bool));
    def("Ones", WRAP_FREE_FUNC_3(Py_DataModeling::Ones3, std::vector<tensorDim_t>, Device, const bool));

    def("Zeros", WRAP_FREE_FUNC_1(Py_DataModeling::Zeros0, std::vector<tensorDim_t>));
    def("Zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros1, std::vector<tensorDim_t>, Device));
    def("Zeros", WRAP_FREE_FUNC_2(Py_DataModeling::Zeros2, std::vector<tensorDim_t>, const bool));
    def("Zeros", WRAP_FREE_FUNC_3(Py_DataModeling::Zeros3, std::vector<tensorDim_t>, Device, const bool));

    def("Gaussian", WRAP_FREE_FUNC_1(Py_DataModeling::Gaussian0, std::vector<tensorDim_t>));
    def("Gaussian", WRAP_FREE_FUNC_2(Py_DataModeling::Gaussian1, std::vector<tensorDim_t>, Device));
    def("Gaussian", WRAP_FREE_FUNC_2(Py_DataModeling::Gaussian2, std::vector<tensorDim_t>, const bool));
    def("Gaussian", WRAP_FREE_FUNC_3(Py_DataModeling::Gaussian3, std::vector<tensorDim_t>, Device, const bool));
}