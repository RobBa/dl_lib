#pragma once

#include "global_params.h"

#include <memory>
#include <type_traits>

class initializer_base {
    public:
        initializer_base() = default;
        virtual ~initializer_base() = default;
        virtual ftype get_random_number() const = 0;
};

class initializer_factory final {
    public:
        initializer_factory() = delete;
        static std::unique_ptr<initializer_base> get_initializer();
};