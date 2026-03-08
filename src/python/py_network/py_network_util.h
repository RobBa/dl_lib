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

#include "layers/layer_base.h"
#include "activation_functions/activation_function_base.h"
#include "training/loss_functions/loss_base.h"
#include "training/optimizers/optimizer_base.h"

#include <boost/python.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>

namespace Py_Network {
  using namespace boost::python;

  ftype layerGetItem(const layers::LayerBase& self, boost::python::object index);
  void layerSetItem(layers::LayerBase& self, boost::python::object index, ftype value);

  /**
   * @brief Wrapper class needed for Boost Python to get the virtual function working 
   * the way it is intended. See documentation here: 
   * https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/exposing.html
   * 
   */
  struct LayerBaseWrap : layers::LayerBase, wrapper<layers::LayerBase> {
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) const override {
      return this->get_override("forward")(input);   
    }

    Tensor forward(const Tensor& input) const override {
      std::__throw_runtime_error("This function should never be called from within Python");   
    }
  };

  struct ActivationFunctionWrap : activation::ActivationFunctionBase, wrapper<activation::ActivationFunctionBase> {
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) const override {
      return this->get_override("call")(input);   
    }
  };

  struct LossWrap : train::LossBase, wrapper<train::LossBase> {
    Tensor operator()(const Tensor& y, const Tensor& ypred) const override {
      return this->get_override("call")(y, ypred);   
    }
  };

  inline std::shared_ptr<Tensor> (LayerBaseWrap::*layerforward)(const std::shared_ptr<Tensor>&) const     = &LayerBaseWrap::forward;
}

