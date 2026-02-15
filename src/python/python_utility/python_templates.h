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

namespace Py_Util {
    /**
     * @brief Convert operator<< to string
     */
    template<typename T>
    std::string toString(const T& obj) {
        std::ostringstream oss;
        oss << obj;
        return oss.str();
    }

    /**
     * @brief Because we manage tensors via shared_ptr, we need this to wrap
     * return values when a function/method demands it.
     */
    /* template<typename Func>
    auto WrapReturnedTensor(Func f) {
        return [f](const Tensor& self, auto&&... args) -> std::shared_ptr<Tensor> {
            return std::make_shared<Tensor>(f(self, std::forward<decltype(args)>(args)...));
        };
    } */
}
