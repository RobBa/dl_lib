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

#include "module/module_base.h"

#include "module/layers/ff_layer.h"

#include "module/activation_functions/relu.h"
#include "module/activation_functions/leaky_relu.h"
#include "module/activation_functions/softmax.h"

#include <boost/python.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>

namespace Py_nn {
  using namespace boost::python;

  /**
   * @brief Wrapper class needed for Boost Python to get the virtual function working 
   * the way it is intended. See documentation here: 
   * https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/exposing.html
   * 
   */
  struct ModuleBaseWrapper : module::ModuleBase, wrapper<module::ModuleBase> {
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) const override {
      return this->get_override("forward")(input);   
    }

    Tensor operator()(const Tensor& input) const override {
      std::__throw_runtime_error("This function should never be called from within Python");   
    }
  };

  inline std::shared_ptr<Tensor> (ModuleBaseWrapper::*moduleForward)(const std::shared_ptr<Tensor>&) const  = &ModuleBaseWrapper::operator();

  inline std::shared_ptr<Tensor> (module::FfLayer::*ffForward)(const std::shared_ptr<Tensor>&) const        = &module::FfLayer::operator();

  inline std::shared_ptr<Tensor> (module::ReLu::*reluF)(const std::shared_ptr<Tensor>&) const               = &module::ReLu::operator();
  inline std::shared_ptr<Tensor> (module::LeakyReLu::*leakyReluF)(const std::shared_ptr<Tensor>&) const     = &module::LeakyReLu::operator();
  inline std::shared_ptr<Tensor> (module::Softmax::*softmaxF)(const std::shared_ptr<Tensor>&) const         = &module::Softmax::operator();
}

