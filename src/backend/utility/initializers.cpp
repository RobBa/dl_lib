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
    private:
      std::random_device rd{};
      mutable std::mt19937 gen;
      mutable std::normal_distribution<ftype> dist;

    public:
        GaussianInitializer(ftype mean, ftype stddev);
        ftype drawNumber() const override;
    };

    GaussianInitializer::GaussianInitializer(ftype mean, ftype stddev) 
      : InitializerBase(), gen{rd()}, dist{mean, stddev} {}

    ftype GaussianInitializer::drawNumber() const {
        return dist(gen);
    }
}

unique_ptr<InitializerBase> InitializerFactory::getInitializer(InitClass ic, ftype mean, ftype stddev) {
    switch(ic){
        case InitClass::Gaussian:
            return make_unique<GaussianInitializer>(mean, stddev);
        default:
            __throw_invalid_argument("Init class not implemented yet");
    }
    return nullptr; // never reached, suppress warning
}