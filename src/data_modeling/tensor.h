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
#include "initializers.h"

#include <memory>
#include <iostream>

#include <concepts>
#include <cassert>

#ifndef NDEBUG
    #include <limits>
#endif // NDEBUG

enum class Device {
    CPU,
    CUDA
};

constexpr const char* DeviceToString(Device d) {
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
         * memory management logic for different devices, like a GPU.
         * Structured as a flat array, the logic for multiple dimensions 
         * encapsulated by surrounding tensor object. 
         */
        struct tensorValues_t final {
        private:
            tensorSize_t size = 0;
            ftype* values = nullptr;
            Device device;

        public:
            tensorValues_t() = delete;
            explicit tensorValues_t(Device d);
            ~tensorValues_t() noexcept;

            tensorValues_t(const tensorValues_t& other) noexcept = delete;
            tensorValues_t& operator=(const tensorValues_t& other) noexcept = delete;

            tensorValues_t(tensorValues_t&& other) noexcept;
            tensorValues_t& operator=(tensorValues_t&& other) noexcept;

            explicit operator bool() const noexcept;
            ftype& operator[](int idx);
            ftype get(int idx) const;

            void setDevice(const Device d) noexcept;
            Device getDevice() const noexcept;

            tensorSize_t getSize() const noexcept;

            template<typename T>
            requires (std::is_integral_v< std::remove_const_t<T> >)
            void resize(const T size) {
                this->size = static_cast<tensorSize_t>(size);

                switch(this->device){
                    case Device::CPU:
                        values = static_cast<ftype*>( std::malloc(size * sizeof(ftype)) );
                        break;
                    case Device::CUDA:
                        std::__throw_invalid_argument("Not implemented yet.");
                        break;
                }
            }

            static tensorValues_t createDeepCopy(const tensorValues_t& v);
        }; 

        std::shared_ptr<tensorValues_t> values = nullptr;

        Dimension dims;
        TensorType type;

#ifndef NDEBUG
        /**
         * @brief Helps us to detect overflows in the folding expression below.
         */
        struct SafeMultiplier_t {
            tensorSize_t value;
            
            SafeMultiplier_t(tensorSize_t v) : value(v) {}
            
            SafeMultiplier_t operator*(const SafeMultiplier_t& other) const {
                if (other.value != 0 && 
                    value > std::numeric_limits<tensorSize_t>::max() / other.value) {
                    throw std::overflow_error("Multiplication overflow");
                }
                return SafeMultiplier_t(value * other.value);
            }
        };
#endif // NDEBUG

        /**
         * @brief Folding expression since C++17: Does the 
         * product of the variadic templated types and returns 
         * them.
         */
        template<typename... T>
        tensorSize_t varProduct(T... x) {
            static_assert(sizeof...(x) >= 2);
#ifndef NDEBUG
            return (SafeMultiplier_t(x) * ...).value; // detects overflows
#else
            return (static_cast<tensorSize_t>(x) * ...); // does not detect overflows
#endif // NDEBUG
        }
    
        Tensor multiplyScalar(const Tensor& scalar, const Tensor& other) const;
        Tensor multiply2D(const Tensor& left, const Tensor& right) const;

        friend void printValuesCpu(std::ostream& os, const Tensor& t);

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

            values = std::make_shared<tensorValues_t>(d);
            values->resize(1);

#ifndef NDEBUG
            std::cout << "Created scalar tensor" << std::endl;
#endif // NDEBUG
        }
        
        template<typename T> requires (is_valid_dim<T>)
        explicit Tensor(T dim1, Device d=Device::CPU) {
            assert(dim1 >= 0);

            type = TensorType::OneD;
            dims[0] = dim1;
            
            values = std::make_shared<tensorValues_t>(d);
            values->resize(dim1);

#ifndef NDEBUG
            std::cout << "Created 1D tensor with dim (" << dim1 << ")" << std::endl;
#endif // NDEBUG
        }

        template<typename T> requires (is_valid_dim<T>)
        explicit Tensor(T dim1, T dim2, Device d=Device::CPU) {
            assert(dim1 >= 0);
            assert(dim2 >= 0);

            type = TensorType::TwoD;
            dims[0] = dim1;
            dims[1] = dim2;

            values = std::make_shared<tensorValues_t>(d);
            values->resize(varProduct(dim1, dim2));

#ifndef NDEBUG
            std::cout << "Created 2D tensor with dims (" << dim1 << "," << dim2 
                << ")" << "\n" << "Dims: (" << dims[0] << "," << dims[1] << ")"
                << "\nSize: " << values->getSize() << std::endl;
#endif // NDEBUG
        }

        template<typename T> requires (is_valid_dim<T>)
        explicit Tensor(T dim1, T dim2, T dim3, Device d=Device::CPU) { 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);

            type = TensorType::ThreeD;
            dims[0] = dim1;
            dims[1] = dim2;
            dims[2] = dim3;

            values = std::make_shared<tensorValues_t>(d);
            values->resize(varProduct(dim1, dim2, dim3));

#ifndef NDEBUG
            std::cout << "Created 3D tensor with dims (" << dim1 << "," << dim2 
                << "," << dim3 << ")" << std::endl;
#endif // NDEBUG
        }

        template<typename T> requires (is_valid_dim<T>)
        explicit Tensor(T dim1, T dim2, T dim3, T dim4, Device d=Device::CPU) { 
            assert(dim1 >= 0);
            assert(dim2 >= 0);
            assert(dim3 >= 0);
            assert(dim4 >= 0);

            type = TensorType::FourD;
            dims[0] = dim1; 
            dims[1] = dim2; 
            dims[2] = dim3; 
            dims[3] = dim4;

            values = std::make_shared<tensorValues_t>(d);
            values->resize(varProduct(dim1, dim2, dim3, dim4));

#ifndef NDEBUG
            std::cout << "Created 4D tensor with dims (" << dim1 << "," << dim2 
                << "," << dim3 << "," << dim4 << ")" << std::endl;
#endif // NDEBUG
        }

        void initialize(const std::unique_ptr<utility::InitializerBase>& init);
        const Dimension& getDims() const noexcept;

        Tensor operator*(Tensor const& t) const;
        static Tensor multiply(const Tensor& left, const Tensor& right);

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) noexcept;

        ftype get(int idx) const;
        ftype get(int idx1, int idx2) const;
        ftype get(int idx1, int idx2, int idx3) const;
        ftype get(int idx1, int idx2, int idx3, int idx4) const;

        void set(ftype item, int idx);
        void set(ftype item, int idx1, int idx2);
        void set(ftype item, int idx1, int idx2, int idx3);
        void set(ftype item, int idx1, int idx2, int idx3, int idx4);
};