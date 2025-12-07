/**
 * @file initializers.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "initializers.h"

#include <random>
#include <algorithm>

using namespace std;
using namespace utility;

namespace {
    class GaussianInitializer final : public InitializerBase {
    public:
        GaussianInitializer();
        ftype getRandomNumber() const override;
    };

    GaussianInitializer::GaussianInitializer() : InitializerBase() {}

    ftype GaussianInitializer::getRandomNumber() const {
        static std::random_device rd;
        static std::mt19937 gen{rd()};
        static std::normal_distribution<ftype> dist;

        return dist(gen);
    }
}

unique_ptr<InitializerBase> InitializerFactory::getInitializer() {
    return make_unique<GaussianInitializer>();
}