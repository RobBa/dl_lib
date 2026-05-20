/**
 * @file main.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-05-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <gtest/gtest.h>

#include "system/sys_functions.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  sys::setRandomSeed(42);
  return RUN_ALL_TESTS();
}