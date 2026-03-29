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

#include "data_modeling/dim_type.h"
#include "utility/initializers.h"

#include "data_modeling/tensor.h"
#include "data_modeling/tensor_functions.h"
#include "computational_graph/tensor_ops/graph_creation.h"

#include <boost/python.hpp>
#include <boost/python/object.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <memory>
#include <stdfloat>

namespace Py_DataModeling
{

  /*********************************************************************************************************
  ********************************************** Dimension *************************************************
  *********************************************************************************************************/

  inline bool (Dimension::*dimEquals1)(const Dimension &) const = &Dimension::operator==;
  inline bool (Dimension::*dimEquals2)(const std::vector<tensorDim_t> &) const = &Dimension::operator==;

  inline bool (Dimension::*nDimEquals1)(const Dimension &) const = &Dimension::operator!=;
  inline bool (Dimension::*nDimEquals2)(const std::vector<tensorDim_t> &) const = &Dimension::operator!=;
  /*********************************************************************************************************
  *********************************************** Tensor ***************************************************
  *********************************************************************************************************/

  ftype tensorGetItem(const Tensor &self, boost::python::object index);
  void tensorSetItem(Tensor &self, boost::python::object index, ftype value);

  template <typename T>
  std::shared_ptr<Tensor> fromNumpy(boost::python::object npArray);

  inline boost::python::object toNumpy(const Tensor& t);

  static bool initNumpy() {
    import_array1(false);  // variant that returns bool
    return true;
  }

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

  inline auto GaussianWrapper0(std::vector<tensorDim_t> dims, ftype stddev) { 
    return TensorFunctions::Gaussian(std::move(dims), stddev);
  }

  inline auto GaussianWrapper1(std::vector<tensorDim_t> dims, Device d, ftype stddev) { 
    return TensorFunctions::Gaussian(std::move(dims), d, stddev);
  }

  inline Tensor    (*Ones0)(std::vector<tensorDim_t>)                                             = &OnesWrapper0;
  inline Tensor    (*Ones1)(std::vector<tensorDim_t>, Device)                                     = &OnesWrapper1;
  inline Tensor    (*Ones2)(std::vector<tensorDim_t>, const bool)                                 = &(TensorFunctions::Ones);
  inline Tensor    (*Ones3)(std::vector<tensorDim_t>, Device, const bool)                         = &(TensorFunctions::Ones);

  inline Tensor    (*Zeros0)(std::vector<tensorDim_t>)                                            = &ZerosWrapper0;
  inline Tensor    (*Zeros1)(std::vector<tensorDim_t>, Device)                                    = &ZerosWrapper1;
  inline Tensor    (*Zeros2)(std::vector<tensorDim_t>, const bool)                                = &(TensorFunctions::Zeros);
  inline Tensor    (*Zeros3)(std::vector<tensorDim_t>, Device, const bool)                        = &(TensorFunctions::Zeros);

  inline Tensor    (*Gaussian0)(std::vector<tensorDim_t>, ftype)                                  = &GaussianWrapper0;
  inline Tensor    (*Gaussian1)(std::vector<tensorDim_t>, Device, ftype)                          = &GaussianWrapper1;
  inline Tensor    (*Gaussian2)(std::vector<tensorDim_t>, ftype, const bool)                      = &(TensorFunctions::Gaussian);
  inline Tensor    (*Gaussian3)(std::vector<tensorDim_t>, Device, ftype, const bool)              = &(TensorFunctions::Gaussian);

  inline void    (Tensor::*reset1)(const ftype)                                                   = &Tensor::reset;
  inline void    (Tensor::*reset2)(const std::shared_ptr<utility::InitializerBase>)               = &Tensor::reset;

  inline Tensor  (Tensor::*transpose1)()                                                          = &Tensor::transpose;
  inline Tensor  (Tensor::*transpose2)(int, int)                                                  = &Tensor::transpose;

  inline ftype   (Tensor::*getItemVector)(const std::vector<tensorDim_t>&) const                  = &Tensor::get;

  /*********************************************************************************************************
  ***************************************** Graph creation *************************************************
  *********************************************************************************************************/

  // multiplications
  inline std::shared_ptr<Tensor> (*elementwisemul) 
  (const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right)           = &(cgraph::mul);

  inline std::shared_ptr<Tensor> (*scalarmul) 
  (const std::shared_ptr<Tensor>, ftype)                                              = &(cgraph::mul);

  inline std::shared_ptr<Tensor> (*rscalarmul) 
  (ftype, const std::shared_ptr<Tensor>)                                              = &(cgraph::mul);

  // additions
  inline std::shared_ptr<Tensor> (*elementwiseadd) 
  (const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right)           = &(cgraph::add);

  inline std::shared_ptr<Tensor> (*scalaradd) 
  (const std::shared_ptr<Tensor>, ftype)                                              = &(cgraph::add);

  inline std::shared_ptr<Tensor> (*rscalaradd) 
  (ftype, const std::shared_ptr<Tensor>)                                              = &(cgraph::add);

  // matmul
  inline std::shared_ptr<Tensor> (*matmul) 
  (const std::shared_ptr<Tensor> left, const std::shared_ptr<Tensor> right)           = &(cgraph::matmul);

  // sub, div
  inline std::shared_ptr<Tensor> (*scalarsub) 
  (const std::shared_ptr<Tensor>, ftype)                                              = &(cgraph::sub);

  inline std::shared_ptr<Tensor> (*scalardiv) 
  (const std::shared_ptr<Tensor>, ftype)                                              = &(cgraph::div);

  // get
  inline std::shared_ptr<Tensor> (*getItemAsTensor1) 
  (const std::shared_ptr<Tensor>& t, tensorSize_t idx)                                = &(cgraph::get);

  inline std::shared_ptr<Tensor> (*getItemAsTensor2) 
  (const std::shared_ptr<Tensor>& t, const std::vector<tensorDim_t>& idx)             = &(cgraph::get);
}


template <typename T>
constexpr int numpyTypeCode()
{
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::float32_t>)
    return NPY_FLOAT32;
  else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::float64_t>)
    return NPY_FLOAT64;
  else if constexpr (std::is_same_v<T, std::float16_t>)
    return NPY_FLOAT16;
  else
    static_assert(false, "Unexpected ftype");
}

template <typename T>
std::shared_ptr<Tensor> Py_DataModeling::fromNumpy(boost::python::object npArray)
{
  using namespace boost::python;

  if (!PyArray_API){
    throw std::runtime_error("Numpy C API not initialized. Call import_array() first.");
  }

  // ensure we have a contiguous float32/double array
  PyObject *obj = npArray.ptr();

  if (!PyArray_Check(obj))
    throw std::invalid_argument("Expected numpy array");

  PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);

  // ensure contiguous C order
  if (!PyArray_IS_C_CONTIGUOUS(arr)){
    arr = reinterpret_cast<PyArrayObject *>(
        PyArray_GETCONTIGUOUS(arr));
  }

  // get shape
  int ndim = PyArray_NDIM(arr);
  std::vector<tensorDim_t> dims(ndim);
  npy_intp *shape = PyArray_SHAPE(arr);
  tensorSize_t totalSize = 1;
  for (int i = 0; i < ndim; i++){
    dims[i] = static_cast<tensorDim_t>(shape[i]);
    totalSize *= dims[i];
  }

  PyArrayObject *floatArr = arr;
  // cast if needed
  if (PyArray_TYPE(arr) != numpyTypeCode<T>()){
    floatArr = reinterpret_cast<PyArrayObject *>(
        PyArray_Cast(arr, numpyTypeCode<T>()));
  }

  const ftype *data = static_cast<const ftype *>(PyArray_DATA(floatArr));
  auto result = std::make_shared<Tensor>(dims, data, totalSize);

  // cleanup if we created new arrays
  if (floatArr != arr)
    Py_DECREF(floatArr);
  if (arr != reinterpret_cast<PyArrayObject *>(obj))
    Py_DECREF(arr);

  return result;
}

boost::python::object Py_DataModeling::toNumpy(const Tensor &t)
{
  using namespace boost::python;

  if (!PyArray_API){
    throw std::runtime_error("Numpy C API not initialized. Call import_array() first.");
  }

  // get shape
  const auto &dims = t.getDims();
  int ndim = static_cast<int>(dims.nDims());

  std::vector<npy_intp> shape(ndim);
  for (int i = 0; i < ndim; i++)
    shape[i] = static_cast<npy_intp>(dims[i]);

  // determine numpy dtype from ftype
  constexpr int dtype = numpyTypeCode<ftype>();

  // allocate numpy array
  PyObject *arr = PyArray_SimpleNew(ndim, shape.data(), dtype);
  if (!arr)
    throw std::runtime_error("Failed to allocate numpy array");

  // copy data
  ftype *dst = static_cast<ftype *>(PyArray_DATA(
      reinterpret_cast<PyArrayObject *>(arr)));

  const tensorSize_t size = t.getSize();
  for (tensorSize_t i = 0; i < size; i++)
    dst[i] = t[i];

  return object(handle<>(arr));
}