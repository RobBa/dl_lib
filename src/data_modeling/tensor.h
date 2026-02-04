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
#include "topological_sort.h"
#include "graph_node.h"

#include "global_params.h"
#include "initializers.h"

#include <memory>
#include <optional>

#include <iostream>

#include <concepts>
#include <cassert>

#ifndef NDEBUG
    #include <limits>
#endif // NDEBUG

// break circular dependency
namespace graph {
    class GraphNode;
    class TopologicalSort;
}

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

class Tensor final {
    enum class TensorType {
        Scalar,
        OneD,
        TwoD,
        ThreeD,
        FourD
    };

    friend class graph::TopologicalSort;

    private:
        /**
         * @brief Here we encapsulate the tensor's values. 
         * Enables us to use a shared_ptr, as well as encapsulate all the 
         * memory management logic for different devices, like a GPU.
         * Structured as a flat array, the logic for multiple dimensions 
         * encapsulated by surrounding tensor object. 
         */
        class tensorValues_t final {
        private:
            tensorSize_t size = 0;
            ftype* values = nullptr;
            
            Device device;
            inline static Device defaultDevice = Device::CPU;

            void addOtherCpu(const tensorValues_t& other) noexcept;

        public:
            explicit tensorValues_t();
            explicit tensorValues_t(Device d);
            ~tensorValues_t() noexcept;

            tensorValues_t(const tensorValues_t& other) = delete;
            tensorValues_t& operator=(const tensorValues_t& other) = delete;

            tensorValues_t(tensorValues_t&& other) noexcept;
            tensorValues_t& operator=(tensorValues_t&& other) noexcept;

            explicit operator bool() const noexcept;
            ftype& operator[](int idx);
            ftype get(int idx) const;

            tensorSize_t getSize() const noexcept;

            // needed for gradient descent
            tensorValues_t& operator+=(const tensorValues_t& other);

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
            
            void setDevice(const Device d) noexcept;
            Device getDevice() const noexcept;

            static void copyValues(tensorValues_t& target, const tensorValues_t& origin);

            static void setDefaultDevice(const Device d) noexcept;
            static Device getDefaultDevice() noexcept;
        };

        bool requiresGrad = false;
        std::optional<Tensor> grads = std::nullopt; // gradients

        std::unique_ptr<tensorValues_t> values = nullptr; // values of tensor TODO: make unique?
        std::shared_ptr<graph::GraphNode> cgNode = nullptr;

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
            static_assert(sizeof...(x) >= 0);
#ifndef NDEBUG
            return (SafeMultiplier_t(x) * ...).value; // detects overflows
#else
            return (static_cast<tensorSize_t>(x) * ...); // does not detect overflows
#endif // NDEBUG
        }
    
        Tensor multiplyScalar(const Tensor& scalar, const Tensor& other) const noexcept;
        Tensor multiply2D(const Tensor& left, const Tensor& right) const;

        friend void printValuesCpu(std::ostream& os, const Tensor& t);

        template<size_t idx, typename First, typename... Rest>
        requires (is_valid_dim<First>)
        void populateDims(First first, Rest... rest) {
            assert(static_cast<tensorDim_t>(first) >= 0);
            dims[idx] = static_cast<tensorDim_t>(first);
            populateDims<idx+1>(rest...);
        }

        // base case
        template<size_t idx>
        void populateDims() {}

        template<typename... T>
        void constructTensor(bool requiresGrad, Device d, T... dimensions) {
            if constexpr(sizeof...(T)==4){
                type = TensorType::FourD;
            }
            else if constexpr(sizeof...(T)==3){
                type = TensorType::ThreeD;
            }
            else if constexpr(sizeof...(T)==2){
                type = TensorType::TwoD;
            }
            else if constexpr(sizeof...(T)==1){
                type = TensorType::OneD;
            }
            else if constexpr(sizeof...(T)==0){
                type = TensorType::Scalar;
            }

            this->requiresGrad = requiresGrad;

            populateDims<0>(dimensions...);

            values = std::make_unique<tensorValues_t>(d);
            if constexpr (sizeof...(T)==0){
                values->resize(1);
            }
            else {
                values->resize(varProduct(dimensions...));
            }
        }

    public:
        template<typename... T>
        explicit Tensor(bool requiresGrad, Device d, T... dimensions) {
            static_assert(sizeof...(dimensions)<=MAX_TENSOR_DIMS, 
                          "Too many dimensions given for ctor");
            
            constructTensor(requiresGrad, d, dimensions...);
        }

        template<typename... T>
        explicit Tensor(T... dimensions) : 
            Tensor(false, tensorValues_t::getDefaultDevice(), dimensions...) 
        {}

        template<typename... T>
        explicit Tensor(bool requiresGrad, T... dimensions) : 
            Tensor(requiresGrad, tensorValues_t::getDefaultDevice(), dimensions...) 
        {}

        template<typename... T>
        explicit Tensor(Device d, T... dimensions) :
            Tensor(false, d, dimensions...)
        {}

        explicit Tensor(bool requiresGrad, Device d, Dimension dim) {
            switch(type){
                case TensorType::FourD:
                    constructTensor(requiresGrad, d, dim.get(0), dim.get(1), dim.get(2), dim.get(3));
                    break;
                case TensorType::ThreeD:
                    constructTensor(requiresGrad, d, dim.get(0), dim.get(1), dim.get(2));
                    break;
                case TensorType::TwoD:
                    constructTensor(requiresGrad, d, dim.get(0), dim.get(1));
                    break;
                case TensorType::OneD:
                    constructTensor(requiresGrad, d, dim.get(0));
                    break;
                case TensorType::Scalar:
                    constructTensor(requiresGrad, d);
                    break;
            }
        }

        /** 
         * Tensors can become very large. Deleting those two
         * helps us to not accidentally copy something we do not 
         * intend to copy. We create extra methods for that.
         */
        Tensor(const Tensor& other) = delete;
        Tensor& operator=(const Tensor& other) = delete;

        Tensor createEmptyCopy() const;
        Tensor createDeepCopy() const;

        /**
         * @brief Moving. Move array of values
         * to new instance as well.
         */
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        void reset(const ftype x);
        void reset(const utility::InitClass ic);
        
        const Dimension& getDims() const noexcept;
        tensorSize_t getSize() const noexcept;

        Tensor operator*(Tensor const& t) const;
        Tensor matMul(const Tensor& other) const;

        Tensor operator+(const Tensor& other) const;
        Tensor add(const Tensor& other) const;

        Tensor elementwiseMul(const Tensor& other) const;

        void backward();

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) noexcept;

        ftype get(int idx) const;
        ftype get(int idx1, int idx2) const;
        ftype get(int idx1, int idx2, int idx3) const;
        ftype get(int idx1, int idx2, int idx3, int idx4) const;

        void set(ftype item, int idx);
        void set(ftype item, int idx1, int idx2);
        void set(ftype item, int idx1, int idx2, int idx3);
        void set(ftype item, int idx1, int idx2, int idx3, int idx4);

        ftype& operator[](const tensorSize_t idx);

        void setDevice(const Device d) noexcept;
        Device getDevice() const noexcept;

        // these two should not be exposed to the python interface
        static void setDefaultDevice(const Device d) noexcept;
        static Device getDefaultDevice() noexcept;
};