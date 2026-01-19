/**
 * @file global_templates.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-01-19
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <sstream>
#include <string>

/**
 * @brief Convert operator<< to string
 */
template<typename T>
std::string toString(const T& obj) {
    std::ostringstream oss;
    oss << obj;
    return oss.str();
}