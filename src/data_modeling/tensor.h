/**
 * @file tensor.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "dim_type.h"
#include "global_params.h"

#include <unordered_map>
#include <memory>

#include <concepts>
#include <cassert>

enum class Device {
    CPU,
    CUDA
};

consteval const char* DeviceToString(Device d) {
    switch(d){
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
    }
}

struct Tensor final {
    enum class TensorType {
        Scalar,
        OneD,
        TwoD,
        ThreeD,
        FourD
    };

    private:
        /**
         * @brief Here we encapsulate the tensor's values. 
         * Enables us to use a shared_ptr, as well as encapsulate all the 
         * logic that could branch out, e.g. memory management through
         * different devices like a GPU.
         * 
         * WARNING: Only Tensor should be seeing this class.
         */
        struct value_t final {
        private:
            tensorSize_t size;
            ftype* values = nullptr;
            Device device;

        public:
            value_t() = delete;
            explicit value_t(Device d);
            ~value_t() noexcept;

            value_t(const value_t& other) noexcept = delete;
            value_t& value(const value_t& other) noexcept = delete;

            value_t(value_t&& other) noexcept;
            value_t& operator=(value_t&& other) noexcept;

            explicit operator bool() const noexcept;
            ftype& operator[](int idx);

            void setDevice(const Device d) noexcept;
            Device getDevice() const noexcept;

            template<typename T>
            requires (std::is_integral_v< std::remove_const_t<T> >)
            void resize(const T size) {
                this->size = static_cast<tensorSize_t>(size);

                switch(this->device){
                    case Device::CPU:
                    values = static_cast<ftype*>( malloc(size * sizeof(ftype)) );
                    break;
                    case Device::CUDA:
                    std::__throw_invalid_argument("Not implemented yet.");
                    break;
                }
            }

            static value_t createDeepCopy(const value_t& other);
        };

        std::shared_ptr<value_t> values = nullptr;

        Dimension dims;
        TensorType type;

        /**
         * @brief Folding expression since C++17: Does the 
         * product of the variadic templated types and returns 
         * them.
         */
        template<typename... T>
        tensorSize_t varProduct(T... x){
            return (static_cast<tensorSize_t>(x) * ...);
        }

        Tensor multiplyScalar(const Tensor& scalar, const Tensor& other) const;
        Tensor multiply2D(const Tensor& left, const Tensor& right) const;

    public:
        /** 
         * @brief Copying. 
         * 
         * WARNING: only copy dimensions and other properties, but
         * underlying array of values shared among the instances via
         * pointer.
         */
        Tensor(const Tensor& other) noexcept;
        Tensor& operator=(const Tensor& other) noexcept;

        /**
         * @brief Moving. Move array of values
         * to new instance as well.
         */
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        explicit Tensor(Device d=Device::CPU) {
            type = TensorType::Scalar;
            dims[0] = 1;

            values = std::make_shared<value_t>(d);
            values->resize(1);
        }
        
        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, Device d=Device::CPU) {
            assert(dim1 >= 0);

            type = TensorType::OneD;
            dims[0] = dim1;
            
            values = std::make_shared<value_t>(d);
            values->resize(dim1);
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, Device d=Device::CPU) {
            assert(dim1 >= 0);
            assert(dim2 >= 0);

            type = TensorType::TwoD;
            dims[0] = dim1;
            dims[1] = dim2;

            values = std::make_shared<value_t>(d);
            values->resize(varProduct(dim1, dim2));
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, T dim3, Device d=Device::CPU) { 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);

            type = TensorType::ThreeD;
            dims[0] = dim1;
            dims[1] = dim2;
            dims[2] = dim3;

            values = std::make_shared<value_t>(d);
            values->resize(varProduct(dim1, dim2, dim3));
        }

        template<typename T> requires (is_valid_dim<T>)
        Tensor(T dim1, T dim2, T dim3, T dim4, Device d=Device::CPU) { 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);
            assert(dim4 >= 0);

            type = TensorType::FourD;
            dims[0] = dim1; 
            dims[1] = dim2; 
            dims[2] = dim3; 
            dims[3] = dim4;

            values = std::make_shared<value_t>(d);
            values->resize(varProduct(dim1, dim2, dim3, dim4));
        }

        const Dimension& getDims() const noexcept;

        Tensor operator*(Tensor const& t) const;

        static Tensor multiply(const Tensor& left, const Tensor& right);
};