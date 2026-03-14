/**
 * @file py_core_util.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-21
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "py_core_util.h"

#include <stdexcept>
#include <utility>

using namespace boost::python;

ftype Py_DataModeling::tensorGetItem(const Tensor& self, boost::python::object index) {
  extract<int> int_extractor(index);
        
  // Single integer index (1D)
  if(int_extractor.check()) {
    auto i0 = static_cast<tensorDim_t>(int_extractor());
    return self.get(i0);
  }
        
  // Tuple index (2D, 3D, or 4D, or list)
  if (PySequence_Check(index.ptr())) {
    int len = PySequence_Length(index.ptr());
        
    // Dispatch to convenience functions for 1-4 args
    if (len == 1) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      return self.get(i0);
    }
    else if (len == 2) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      auto i1 = static_cast<tensorDim_t>(extract<int>(index[1]));
      return self.get(i0, i1);
    }
    else if (len == 3) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      auto i1 = static_cast<tensorDim_t>(extract<int>(index[1]));
      auto i2 = static_cast<tensorDim_t>(extract<int>(index[2]));
      return self.get(i0, i1, i2);
    }
    else if (len == 4) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      auto i1 = static_cast<tensorDim_t>(extract<int>(index[1]));
      auto i2 = static_cast<tensorDim_t>(extract<int>(index[2]));
      auto i3 = static_cast<tensorDim_t>(extract<int>(index[3]));
      return self.get(i0, i1, i2, i3);
    }
    else {
      // Arbitrary length - use vector version
      std::vector<tensorDim_t> indices;
      for (int i = 0; i < len; ++i) {
        indices.push_back(static_cast<tensorDim_t>(extract<int>(index[i])));
      }
      return self.get(std::move(indices));
    }
  }
        
  PyErr_SetString(PyExc_TypeError, "Index must be a number of up to 4integers or a list");
  throw_error_already_set();
  return 0.0; // Never reached
}

void Py_DataModeling::tensorSetItem(Tensor& self, boost::python::object index, ftype value) {
  extract<int> int_extractor(index);
  if(int_extractor.check()) {
      auto i0 = static_cast<tensorDim_t>(int_extractor());
      self.set(value, i0);
      return;
  }
        
  // Tuple index (2D, 3D, or 4D, or list)
  extract<tuple> tuple_extractor(index);
  if (PySequence_Check(index.ptr())) {
    int len = PySequence_Length(index.ptr());
        
    // Dispatch to convenience functions for 1-4 args
    if (len == 1) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      self.set(value, i0);
    }
    else if (len == 2) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      auto i1 = static_cast<tensorDim_t>(extract<int>(index[1]));
      self.set(value, i0, i1);
    }
    else if (len == 3) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      auto i1 = static_cast<tensorDim_t>(extract<int>(index[1]));
      auto i2 = static_cast<tensorDim_t>(extract<int>(index[2]));
      self.set(value, i0, i1, i2);
    }
    else if (len == 4) {
      auto i0 = static_cast<tensorDim_t>(extract<int>(index[0]));
      auto i1 = static_cast<tensorDim_t>(extract<int>(index[1]));
      auto i2 = static_cast<tensorDim_t>(extract<int>(index[2]));
      auto i3 = static_cast<tensorDim_t>(extract<int>(index[3]));
      self.set(value, i0, i1, i2, i3);
    }
    else {
      // Arbitrary length - use vector version
      std::vector<tensorDim_t> indices;
      for (int i = 0; i < len; ++i) {
        indices.push_back(static_cast<tensorDim_t>(extract<int>(index[i])));
      }
      self.set(value, std::move(indices));
    }
    return;
  }
        
  PyErr_SetString(PyExc_TypeError, "Index must be a number of up to 4integers or a list");
  throw_error_already_set();
}