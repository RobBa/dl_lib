#include "initializers.h"

#include <random>
#include <algorithm>

using namespace std;

gaussian_initializer::gaussian_initializer() : initializer_base() {}

double gaussian_initializer::get_random_number() const {
    static std::random_device rd;
    static std::mt19937 gen{rd()};
    static std::normal_distribution<double> dist;

    return dist(gen);
}

unique_ptr<initializer_base> initializer_factory::get_initializer() {
    return make_unique<gaussian_initializer>();
}