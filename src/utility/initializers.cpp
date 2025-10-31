#include "initializers.h"

#include <random>
#include <algorithm>

using namespace std;

gaussian_initializer::gaussian_initializer() : initializer_base() {}

double gaussian_initializer::get_random_number() const {
    static std::random_device rnd_device;
    static std::mt19937 mersenne_engine{rnd_device};
    static std::normal_distribution<double> dist;

    return dist(mersenne_engine);
}

unique_ptr<initializer_base> initializer_factory::get_initializer() {
    return make_unique<gaussian_initializer>();
}