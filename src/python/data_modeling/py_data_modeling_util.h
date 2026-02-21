/**
 * @file util.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Helper and wrapper functions
 * @version 0.1
 * @date 2026-02-21
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"
#include "computational_graph/graph_creation.h"

#include <boost/python.hpp>
#include <boost/python/object.hpp>

#include <memory>

namespace Py_DataModeling {
    ftype tensorGetItem(const Tensor& self, boost::python::object index);
    void tensorSetItem(Tensor& self, boost::python::object index, ftype value);

    // need wrappers for default arguments, see
    // https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/functions.html
    inline auto OnesWrapper0(std::vector<tensorDim_t> dims) { 
        return TensorFunctions::Ones(std::move(dims)); 
    }

    inline auto OnesWrapper1(std::vector<tensorDim_t> dims, Device d) { 
        return TensorFunctions::Ones(std::move(dims), d); 
    }

    inline auto ZerosWrapper0(std::vector<tensorDim_t> dims) { 
        return TensorFunctions::Zeros(std::move(dims)); 
    }

    inline auto ZerosWrapper1(std::vector<tensorDim_t> dims, Device d) { 
        return TensorFunctions::Zeros(std::move(dims), d); 
    }

    inline auto GaussianWrapper0(std::vector<tensorDim_t> dims) { 
        return TensorFunctions::Gaussian(std::move(dims)); 
    }

    inline auto GaussianWrapper1(std::vector<tensorDim_t> dims, Device d) { 
        return TensorFunctions::Gaussian(std::move(dims), d); 
    }

    inline Tensor    (*Ones0)(std::vector<tensorDim_t>)                                 = &OnesWrapper0;
    inline Tensor    (*Ones1)(std::vector<tensorDim_t>, Device)                         = &OnesWrapper1;
    inline Tensor    (*Ones2)(std::vector<tensorDim_t>, const bool)                     = &(TensorFunctions::Ones);
    inline Tensor    (*Ones3)(std::vector<tensorDim_t>, Device, const bool)             = &(TensorFunctions::Ones);

    inline Tensor    (*Zeros0)(std::vector<tensorDim_t>)                                = &ZerosWrapper0;
    inline Tensor    (*Zeros1)(std::vector<tensorDim_t>, Device)                        = &ZerosWrapper1;
    inline Tensor    (*Zeros2)(std::vector<tensorDim_t>, const bool)                    = &(TensorFunctions::Zeros);
    inline Tensor    (*Zeros3)(std::vector<tensorDim_t>, Device, const bool)            = &(TensorFunctions::Zeros);

    inline Tensor    (*Gaussian0)(std::vector<tensorDim_t>)                             = &GaussianWrapper0;
    inline Tensor    (*Gaussian1)(std::vector<tensorDim_t>, Device)                     = &GaussianWrapper1;
    inline Tensor    (*Gaussian2)(std::vector<tensorDim_t>, const bool)                 = &(TensorFunctions::Gaussian);
    inline Tensor    (*Gaussian3)(std::vector<tensorDim_t>, Device, const bool)         = &(TensorFunctions::Gaussian);

    inline void    (Tensor::*reset1)(const ftype)                                       = &Tensor::reset;
    inline void    (Tensor::*reset2)(const utility::InitClass)                          = &Tensor::reset;

    inline void    (Tensor::*transposeThis1)()                                          = &Tensor::transposeThis;
    inline void    (Tensor::*transposeThis2)(int, int)                                  = &Tensor::transposeThis;
    inline Tensor  (Tensor::*transpose1)(int, int) const                                = &Tensor::transpose;
    inline Tensor  (Tensor::*transpose2)(int, int, bool) const                          = &Tensor::transpose;

    inline std::shared_ptr<Tensor> (*getItemAsTensor1) 
    (const std::shared_ptr<Tensor>& t, tensorSize_t idx)                         = &(graph::get);
    
    inline std::shared_ptr<Tensor> (*getItemAsTensor2) 
    (const std::shared_ptr<Tensor>& t, const std::vector<tensorDim_t>& idx)      = &(graph::get);
}