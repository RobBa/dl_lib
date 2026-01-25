/**
 * @file tensor.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "py_data_modeling.h"

#include <stdexcept>

using namespace boost::python;

/**
 * @brief We need these pointers to wrap overloading
 */
ftype (Tensor::*get1)(int)                const = &Tensor::get;
ftype (Tensor::*get2)(int, int)           const = &Tensor::get;
ftype (Tensor::*get3)(int, int, int)      const = &Tensor::get;
ftype (Tensor::*get4)(int, int, int, int) const = &Tensor::get;

ftype Py_DataModeling::get_item(boost::python::object index) {
  extract<int> int_extractor(index);

  // TODO: what to do about no argument at all?
        
  // Single integer index (1D)
  if(int_extractor.check()) {
      int i0 = int_extractor();
      return (*get1)(i0);
  }
        
  // Tuple index (2D, 3D, or 4D)
  extract<tuple> tuple_extractor(index);
  if(tuple_extractor.check()) {
      tuple idx_tuple = tuple_extractor();
      int ndim = boost::python::len(idx_tuple);
      
      if (ndim == 2) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        return (*get2)(i0, i1);
      }
      else if (ndim == 3) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        return (*get3)(i0, i1, i2);
      }
      else if (ndim == 4) {
        int i0 = extract<int>(idx_tuple[0]);
        int i1 = extract<int>(idx_tuple[1]);
        int i2 = extract<int>(idx_tuple[2]);
        int i3 = extract<int>(idx_tuple[3]);
        return (*get4)(i0, i1, i2, i3);
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