/**
 * @file layers.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-11-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "py_layers.h"

#include <stdexcept>

using namespace boost::python;

ftype Py_Layers::layerGetItem(const layers::LayerBase& self, boost::python::object index) {
  extract<int> int_extractor(index);
        
  // Single integer index (1D)
  if(int_extractor.check()) {
      int i0 = int_extractor();
      return self.get(i0);
  }
        
  // Tuple index (2D, 3D, or 4D)
  extract<tuple> tuple_extractor(index);
  if(tuple_extractor.check()) {
      tuple idx_tuple = tuple_extractor();
      int ndim = boost::python::len(idx_tuple);
      
      if (ndim == 2) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        return self.get(i0, i1);
      }
      else if (ndim == 3) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        return self.get(i0, i1, i2);
      }
      else if (ndim == 4) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        int i3 = extract<int>(idx_tuple[3]);
        return self.get(i0, i1, i2, i3);
      }
      else {
        PyErr_SetString(PyExc_IndexError, "Unsupported number of dimensions");
        throw_error_already_set();
      }
  }
        
  PyErr_SetString(PyExc_TypeError, "Index must be an integer or tuple");
  throw_error_already_set();
  return 0.0; // Never reached
}

void Py_Layers::layerSetItem(layers::LayerBase& self, boost::python::object index, ftype value) {
  extract<int> int_extractor(index);
        
  // Single integer index (1D)
  if(int_extractor.check()) {
      int i0 = int_extractor();
      self.set(value, i0);\
      return;
  }
        
  // Tuple index (2D, 3D, or 4D)
  extract<tuple> tuple_extractor(index);
  if(tuple_extractor.check()) {
      tuple idx_tuple = tuple_extractor();
      int ndim = boost::python::len(idx_tuple);
      
      if (ndim == 2) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        self.set(value, i0, i1);
      }
      else if (ndim == 3) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        self.set(value, i0, i1, i2);
      }
      else if (ndim == 4) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        int i3 = extract<int>(idx_tuple[3]);
        self.set(value, i0, i1, i2, i3);
      }
      else {
        PyErr_SetString(PyExc_IndexError, "Unsupported number of dimensions");
        throw_error_already_set();
      }
      return;
  }
        
  PyErr_SetString(PyExc_TypeError, "Index must be an integer or tuple");
  throw_error_already_set();
}