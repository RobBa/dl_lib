/**
 * @file activation_function_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "activation_function_base.h"

using namespace std;
using namespace activation;

ostream& operator<<(ostream& os, const ActivationFunctionBase& l) noexcept {
  static_cast<const ActivationFunctionBase*>(&l)->print(os); // calling vtable
  return os;
}