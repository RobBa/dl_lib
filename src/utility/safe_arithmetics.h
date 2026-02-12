/**
 * @file safe_arithmetics.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-02-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifndef NDEBUG

#include <stdexcept>
#include <limits>

namespace utility {
  /**
   * @brief Helps us to detect overflows in the folding expression below.
   */
  template<typename T>
  struct SafeArithmetics_t {
    T value;
    
    explicit SafeArithmetics_t(T v) : value(v) {}
    
    SafeArithmetics_t operator*(const SafeArithmetics_t& other) const {
      if (other.value != 0 && 
        value > std::numeric_limits<T>::max() / other.value) {
        throw std::overflow_error("Multiplication overflow");
      }
      return SafeArithmetics_t(value * other.value);
    }

    SafeArithmetics_t operator+(const SafeArithmetics_t& other) const {
      if (other.value != 0 && 
        value > std::numeric_limits<T>::max() - other.value) {
        throw std::overflow_error("Addition overflow");
      }
      return SafeArithmetics_t(value + other.value);
    }
  };
}

#endif // NDEBUG