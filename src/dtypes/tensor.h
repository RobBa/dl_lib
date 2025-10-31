#pragma once

template<typename T>
class tensor_2d final {
    private:
        T* values = nullptr;

    public:
        tensor_2d() = default;
        tensor_2d(size_t size);
        tensor_2d()
};