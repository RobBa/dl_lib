/**
 * @file initializers.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "global_params.h"

#include <memory>
#include <type_traits>

namespace utility{
    enum class InitClass {
        Gaussian    
    };

    class InitializerBase {
        public:
            InitializerBase() = default;
            virtual ~InitializerBase() = default;
            virtual ftype drawNumber() const = 0;
    };

    class InitializerFactory final {
        public:
            InitializerFactory() = delete;
            static std::unique_ptr<InitializerBase> getInitializer(InitClass ic);
    };
}
