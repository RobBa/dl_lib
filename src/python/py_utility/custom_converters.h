/**
 * @file custom_converters.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <boost/python.hpp>
#include <boost/python/object.hpp>

#include <type_traits>

#include <vector>
#include <limits>

namespace custom_converters {
  /**
   * @brief We use this class to convert Python lists of int into vectors of
   * internal types, such as tensorDim_t.
   */
  template<typename T>
  requires ( std::is_integral_v< T > || 
             std::is_floating_point_v< T >)
  struct PyListToVectorConverter {
    using rvalueFromPythonData = boost::python::converter::rvalue_from_python_stage1_data;

    PyListToVectorConverter();

    static void* convertible(PyObject* obj_ptr);
    static void construct(PyObject* obj_ptr,rvalueFromPythonData* data);
  };

  /**
   * @brief We use this class to convert Python integers into
   * internal types, such as tensorDim_t.
   */
  template<typename T>
  requires ( std::is_integral_v< T >)
  struct PyIntToIntegralValueConverter {
    using rvalueFromPythonData = boost::python::converter::rvalue_from_python_stage1_data;

    PyIntToIntegralValueConverter();
    
    static void* convertible(PyObject* obj_ptr);
    static void construct(PyObject* obj_ptr,rvalueFromPythonData* data);
  };
}

// TODO: do array instead of tensor
/* struct DimsFromPython {
    static void* convertible(PyObject* obj) {
        if (!PyTuple_Check(obj) && !PyList_Check(obj)) return nullptr;
        return obj;
    }
    
    static void construct(PyObject* obj, 
                         bp::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((bp::converter::rvalue_from_python_object_data<Dims>*)data)->storage.bytes;
        Dims* dims = new (storage) Dims();
        int len = PySequence_Length(obj);
        dims->ndim = len;
        for (int i = 0; i < len; i++)
            dims->data[i] = bp::extract<size_t>(PySequence_GetItem(obj, i));
        data->convertible = storage;
    }
};

// register it in your module init:
bp::converter::registry::push_back(
    &DimsFromPython::convertible,
    &DimsFromPython::construct,
    bp::type_id<Dims>()); */

/******************************************************************************************/
/******************************************************************************************/
/******************************************************************************************/

template<typename T>
requires ( std::is_integral_v< T > || 
           std::is_floating_point_v< T >)
custom_converters::PyListToVectorConverter<T>::PyListToVectorConverter() {
  using namespace boost::python;

  // register converter with Boost.Python's conversion system
  converter::registry::push_back(
    &convertible, // Python object convertable?
    &construct, // How to convert
    type_id<std::vector<T>>() // C++ dtype
  ); 
}

template<typename T>
requires ( std::is_integral_v< T > || 
           std::is_floating_point_v< T >)
void* custom_converters::PyListToVectorConverter<T>::convertible(PyObject* obj_ptr) {
  using namespace boost::python;
  
  if (!PySequence_Check(obj_ptr)) 
    return nullptr; 
  
  return obj_ptr;
}

template<typename T>
requires ( std::is_integral_v< T > || 
           std::is_floating_point_v< T >)
void custom_converters::PyListToVectorConverter<T>::construct(PyObject* obj_ptr, rvalueFromPythonData* data) {

  using namespace boost::python;

  // Wrap the Python object
  list py_list(handle<>(borrowed(obj_ptr)));
        
  // Get memory location where C++ object should be constructed
  void* storage = ((converter::rvalue_from_python_storage< std::vector<T> >*)data)->storage.bytes;
        
  // Construct the vector in-place at that location
  std::vector<T>* vec = new (storage) std::vector<T>();
        
  // Fill it with converted values
  for (int i = 0; i < len(py_list); ++i) {

    if constexpr(std::is_integral_v< T >){
      auto val = extract<int>(py_list[i]);
      vec->push_back(static_cast<T>(val));
    }
    else if constexpr(std::is_floating_point_v< T >) {
      auto val = extract<ftype>(py_list[i]);
      vec->push_back(static_cast<T>(val));
    }
  }
        
  // Tell Boost.Python where the constructed object is
  data->convertible = storage;
}

template<typename T>
requires ( std::is_integral_v< T >)
custom_converters::PyIntToIntegralValueConverter<T>::PyIntToIntegralValueConverter() {
  using namespace boost::python;

  // register converter with Boost.Python's conversion system
  converter::registry::push_back(
    &convertible, // Python object convertable?
    &construct, // How to convert
    type_id<T>() // C++ dtype
  ); 
}

template<typename T>
requires ( std::is_integral_v< T >)
void* custom_converters::PyIntToIntegralValueConverter<T>::convertible(PyObject* obj_ptr) {
  using namespace boost::python;

  if (!PyLong_Check(obj_ptr)) 
    return nullptr;
  
  return obj_ptr;
}

template<typename T>
requires ( std::is_integral_v< T >)
void custom_converters::PyIntToIntegralValueConverter<T>::construct(PyObject* obj_ptr, rvalueFromPythonData* data) {
  using namespace boost::python;

  // Extract Python int
  long val = PyLong_AsLong(obj_ptr);
        
  // Range check
  if (val < 0 || val > std::numeric_limits<T>::max()) {
    PyErr_SetString(PyExc_ValueError, "Value out of range for T");
    throw_error_already_set();
  }
        
  // Get storage location
  void* storage = ((converter::rvalue_from_python_storage<T>*)data)->storage.bytes;
        
  // Construct in place
  new (storage) T(static_cast<T>(val));
        
  data->convertible = storage;
}