/**
 * @file module.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-03-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "module/module_base.h"

#include <utility>

using namespace std;

ostream& module::operator<<(ostream& os, const module::ModuleBase& l) noexcept {
  l.print(os); // calling vtable
  return os;
}