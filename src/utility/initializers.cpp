#include "initializers.h"

#include <random>
#include <algorithm>

using namespace std;

namespace {
    class gaussian_initializer final : public initializer_base {
    public:
        gaussian_initializer();
        ftype get_random_number() const override;
    };

    gaussian_initializer::gaussian_initializer() : initializer_base() {}

    ftype gaussian_initializer::get_random_number() const {
        // TODO: optimize those objects so they don't get reinitialized each time
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::normal_distribution<ftype> dist;

        return dist(gen);
    }
}

unique_ptr<initializer_base> initializer_factory::get_initializer() {
    return make_unique<gaussian_initializer>();
}